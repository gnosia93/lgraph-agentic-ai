
## [LeaderWorkerSet](https://lws.sigs.k8s.io/docs/overview/) ##

LLM 서빙이 본격적으로 프로덕션에 들어오면서, 기존 Kubernetes 워크로드 리소스만으로는 다루기 어려운 영역이 생겼다. 대표적인 사례가 모델을 여러 노드에 쪼개서 서빙해야 하는 경우다. Llama 3.1 405B, DeepSeek V3 같은 초대형 모델은 단일 노드의 GPU 메모리로는 감당이 안 된다. H100 80GB 8장짜리 노드(640 GiB) 두 대를 묶어 텐서 병렬(TP)과 파이프라인 병렬(PP)을 조합해야 겨우 돌아간다.

메모리 계산
```
1 B  = 10⁹  = 1,000,000,000         (10억)
405B = 405 × 10⁹ = 405,000,000,000  (4,050억 파라미터)
```
FP16 가중치 메모리:
* 2 bytes × 405 × 10⁹ = 810 × 10⁹ bytes = 810 GB ≈ 754 GiB

여기에 KV 캐시와 활성화 메모리까지 더하면 실서빙에는 800 GiB 이상이 필요하다. p5.48xlarge(H100 8EA) 노드 한 대(640 GiB)로는 가중치조차 올릴 수 없고, 두 대를 묶어야 비로소 여유가 생긴다.

### Deployment와 StatefulSet의 한계점 ###
Deployment와 StatefulSet은 파드에 문제가 생기면 해당 파드만 재시작한다. 단일 파드로 배포되는 워크로드에는 자연스러운 동작이지만, 멀티노드 GPU 서빙에는 적합하지 않다.
TP+PP 구조에서는 여러 파드가 NCCL로 서로 통신하며 하나의 모델을 함께 서빙한다. 파드 하나가 죽으면 NCCL 통신 그룹 전체가 깨지기 때문에, 그룹 내 모든 파드를 함께 재시작해야 한다. Deployment와 StatefulSet은 이와 같은 "그룹 단위 생명주기"라는 개념을 지원하지 않는다.  
그래서 이 공백을 메우기 위해 Kubernetes SIG가 새롭게 도입한 리소스가 바로 LeaderWorkerSet(LWS) 이다.

### LWS 그룹 ###
LWS의 가장 중요한 개념은 그룹이다. 한 그룹은 리더 1개와 워커 N개로 구성되며, LWS의 리프릴카는 파드수가 아니라 그룹수를 의미한다.
```
LeaderWorkerSet (replicas: 2)
├── Group 0
│   ├── Leader Pod  (rank 0)
│   ├── Worker Pod  (rank 1)
│   ├── Worker Pod  (rank 2)
│   └── Worker Pod  (rank 3)
└── Group 1
    ├── Leader Pod  (rank 0)
    ├── Worker Pod  (rank 1)
    ├── Worker Pod  (rank 2)
    └── Worker Pod  (rank 3)
```
그룹 하나가 독립된 추론 단위로, 외부에서 보면 하나의 그룹이 Deployment의 파드처럼 동작한다. 트래픽은 리더 파드로만 들어가고, 그룹을 구성하는 워커들과 협력해 응답을 만들어 낸다.
* Leader: 외부 요청을 받고, 텐서/파이프라인 병렬 연산을 조정한다. (rank 0)
* Worker: 모델 가중치의 일부만 들고 있으며, Leader의 지시를 받아 연산한다.

LWS 컨트롤러는 각 Pod에 필요한 환경변수를 자동으로 주입해 준다. 애플리케이션은 이 값을 읽어서 torchrun의 --master_addr, --rank 같은 파라미터로 넘겨주게 된다.
* LWS_LEADER_ADDRESS: 같은 Group의 Leader Pod DNS 이름
* LWS_WORKER_INDEX: Group 내 rank (0 = leader)
* LWS_GROUP_SIZE: Group 내 Pod 총 수

그룹마다 헤드리스 서비스가 자동으로 만들어지고, 파드들은 안정적인 DNS 이름으로 서로를 찾게 되고 별도의 Service YAML을 필요치 않는다.
startupPolicy를 LeaderCreated로 설정하면 Leader가 먼저 Ready된 뒤에 Worker가 동작하고, master가 먼저 대기 상태로 진입한 후 worker가 Join 한다. 

### 설치하기 ###
아래 manifest YAML 를 사용하여 CRD 와 컨트롤러를 설치한다.
```
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/lws/releases/latest/download/manifests.yaml
```
설치된 오브젝트를 확인한다.
```
kubectl get all -n lws-system
```
[결과]
```
NAME                                          READY   STATUS    RESTARTS   AGE
pod/lws-controller-manager-567cc75d78-4t9js   1/1     Running   0          65s
pod/lws-controller-manager-567cc75d78-cwgxv   1/1     Running   0          65s

NAME                                             TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
service/lws-controller-manager-metrics-service   ClusterIP   172.20.106.195   <none>        8443/TCP   66s
service/lws-webhook-service                      ClusterIP   172.20.78.73     <none>        443/TCP    66s

NAME                                     READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/lws-controller-manager   2/2     2            2           65s

NAME                                                DESIRED   CURRENT   READY   AGE
replicaset.apps/lws-controller-manager-567cc75d78   2         2         2       65s
```

## EFA 확인 및 fsx 드라이버(pv/pvc) 설치 ##





## Llama 3.1 405B 배포하기 ##
다음은 vLLM으로 Llama 3.1 405B를 2노드에 걸쳐 서빙하는 예제이다.

> [!IMPORTANT]
> 본 튜토리얼에서는 GPU 수급 불균형으로 인해 NVLink 지원 인스턴스(p5 이상) 확보가 어려워 g7e.48xlarge(GTX 6000 × 8, PCIe)를 사용한다. 실제 운영 환경에서는 NVLink가 지원되는 p5.48xlarge(H100 × 8) 이상의 GPU 를 선택해야 한다. NVLink는 PCIe 대비 약 7~14배 높은 GPU 간 대역폭을 제공하여, 멀티 GPU 추론 시 통신 병목을 크게 줄여준다.

### 배포 전 확인사항 ###
* hf-token Secret 생성됨 (kubectl create secret generic hf-token --from-literal=token=hf_xxx)
* llama-405b-cache PVC 준비됨 (최소 1TB, FSx Lustre 강력 추천)
* g7e.48xlarge 또는 p5.48xlarge 노드 2대 확보 (NVLink 를 지원하는 p5 시리즈 선호)
* 두 노드 같은 placement group 소속
* EFA device plugin 설치됨
*  NVIDIA device plugin 설치됨

llm-serving 네임스페이스를 먼저 생성한다.
```
kubectl create ns llm-serving
```
LeaderWorkerSet 을 생성한다.
```
cat <<'EOF' | kubectl apply -f - 
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-llama-405b
  namespace: llm-serving
  annotations:
    leaderworkerset.sigs.k8s.io/exclusive-topology: topology.kubernetes.io/zone
spec:
  replicas: 1
  rolloutStrategy:
    type: RollingUpdate
    rollingUpdateConfiguration:
      maxUnavailable: 0
      maxSurge: 1                                # 업데이트 중에 replicas를 초과해서 임시로 추가 생성할 수 있는 그룹의 최대 개수
  leaderWorkerTemplate:
    size: 2                                      # Leader 1, Worker 1 
    restartPolicy: RecreateGroupOnPodRestart     # 그룹 안의 파드 하나가 죽으면, 그룹 전체를 재생성하라 는 정책
    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        terminationGracePeriodSeconds: 120       # 파드를 종료할 때 얼마나 기다려줄지를 초 단위로 지정합니다. 여기선 120초 (2분).
        nodeSelector:
          node.kubernetes.io/instance-type: g7e.48xlarge
        containers:
          - name: vllm-leader
            image: vllm/vllm-openai:v0.6.3
            command: ["/bin/bash", "-c"]
            args:
              - |
                bash /vllm-workspace/ray_init.sh leader \
                  --ray_cluster_size=$LWS_GROUP_SIZE;
                python -m vllm.entrypoints.openai.api_server \
                  --model=meta-llama/Llama-3.1-405B-Instruct \
                  --tensor-parallel-size=8 \
                  --pipeline-parallel-size=2 \
                  --distributed-executor-backend=ray
            env:
              - name: HUGGING_FACE_HUB_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token
                    key: token
              - name: HF_HOME
                value: /models
            resources:
              limits:
                nvidia.com/gpu: 8
 #               vpc.amazonaws.com/efa: 32
 #               memory: 1000Gi
 #               cpu: "48"
            ports:
              - containerPort: 8000
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 900
              periodSeconds: 30
            readinessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 900
              periodSeconds: 10
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
              - name: model-cache
                mountPath: /models
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 32Gi
          - name: model-cache
            persistentVolumeClaim:
              claimName: llama-405b-cache
    workerTemplate:
      spec:
        terminationGracePeriodSeconds: 120
        nodeSelector:
          node.kubernetes.io/instance-type: g7e.48xlarge
        containers:
          - name: vllm-worker
            image: vllm/vllm-openai:v0.6.3
            command: ["/bin/bash", "-c"]
            args:
              - |
                bash /vllm-workspace/ray_init.sh worker \
                  --ray_address=$LWS_LEADER_ADDRESS
            env:
              - name: HUGGING_FACE_HUB_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token
                    key: token
              - name: HF_HOME
                value: /models
            resources:
              limits:
                nvidia.com/gpu: 8
#                vpc.amazonaws.com/efa: 32
#                memory: 1000Gi
#                cpu: "48"
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
              - name: model-cache
                mountPath: /models
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 32Gi
          - name: model-cache
            persistentVolumeClaim:
              claimName: llama-405b-cache
EOF
```
눈여겨 볼점은 리더와 워커의 `명령어셋이 다르다는 점`으로, 리더는 Ray 클러스터를 시작하고 vLLM 서버를 띄우고, 워커는 리더가 띄운 Ray에 조인만 한다.

#### 리더가 하는 일 ####
```
bash ray_init.sh leader --ray_cluster_size=$LWS_GROUP_SIZE   # ① Ray 헤드 노드 시작
python -m vllm...api_server --tensor-parallel-size=8 \        # ② vLLM 서버 시작
  --pipeline-parallel-size=2 --distributed-executor-backend=ray
```
① 먼저 Ray 클러스터의 헤드(head) 노드를 띄운다. ② 그다음 그 위에서 vLLM API 서버를 실행한다.

#### 워커가 하는 일 ####
```
bash ray_init.sh worker --ray_address=$LWS_LEADER_ADDRESS   # Ray 워커로 헤드에 합류
```
리더(헤드)의 주소로 Ray 클러스터에 워커 노드로 참여만 합니다. vLLM 서버는 띄우지 않는다.

여기서 중요한 게 --distributed-executor-backend=ray 파라미터이다. vLLM은 "여러 노드에 vLLM을 여러 개 띄워서" 분산하는 게 아니라, vLLM 서버는 논리적으로 딱 하나이고, 그 하나가 Ray를 통해 클러스터 전체의 GPU에 모델을 펼쳐서 실행하게 된다.
vLLM 서버(API 엔드포인트, 요청 스케줄링, 전체 오케스트레이션) 는 리더(Ray 헤드)에서 실행되고, 실제 모델 연산(GPU 계산): 리더 + 워커의 모든 GPU에 분산되어 실행되게 된다.(Ray가 워커 GPU로 작업을 보냄)

이 매니페스트는 tensor-parallel-size=8, pipeline-parallel-size=2로 TP 8 × PP 2 = 16개 GPU가 필요한데, 노드 하나가 GPU 8개니까 2개 노드(리더 1 + 워커 1, size: 2)가 묶여야 16개가 된다. 리더의 vLLM이 Ray를 통해 이 16개 GPU(리더 8 + 워커 8)를 전부 끌어다 모델 하나를 돌리는 것이다.

요청 흐름을 보면, 클라이언트는 리더의 8000 포트(vLLM API)로만 요청을 보내고, 리더의 vLLM이 Ray를 통해 워커 GPU까지 동원해 추론한 뒤 응답을 돌려둥가.(워커로는 직접 요청이 가지 않는다)
이게 바로 앞서 설명한 LWS의 "리더 = 진입점, 워커 = 연산 담당"이라는 역할 분리가 실제 vLLM/Ray 구성에서 구현된 모습이다.

### vllm/vllm-openai 와 EFA ###

vllm/vllm-openai 이미지는 AWS EFA를 지원하지 않는다. EFA를 쓰려면 컨테이너 안에 EFA 통신 스택이 들어 있어야 하고, 다음 세 가지가 설정되어 있어야 한다.
* libfabric (EFA 프로바이더) — EFA 하드웨어와 통신하는 라이브러리
* aws-ofi-nccl 플러그인 — NCCL이 libfabric/EFA를 통해 통신하도록 이어주는 다리
* NCCL이 위 둘을 쓰도록 된 환경

그런데 공식 vLLM 이미지는 범용 이미지라 이 EFA 스택(특히 aws-ofi-nccl 플러그인)을 포함하지 않다. 그래서 이 이미지를 그대로 쓰면 NCCL이 EFA를 못 찾고 일반 TCP(ENI)로 폴백하게된다. 
EFA를 실제로 쓰려면 vllm/vllm-openai 이미지를 AWS 가 제공하는 vLLM DLC로 교체해야 한다. 해당 이미지는 https://aws.github.io/deep-learning-containers/vllm/ 에서 조회 가능하다.


#### 참고자료 ####
* [vLLM Deepseek Model on EKS with GPU Support, EFA, and FSx Lustre Integration](https://aws.github.io/deep-learning-containers/tutorials/vllm-samples/deepseek/eks/)
* [Deploy LLMs on Amazon EKS using vLLM Deep Learning Containers](https://aws.amazon.com/ko/blogs/architecture/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)
* [Amazon EKS에서vLLM Deep Learning Container를 사용하여LLM 배포하기](https://aws.amazon.com/ko/blogs/tech/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)
* https://github.com/aws-samples/sample-aws-deep-learning-containers/blob/main/vllm-samples/deepseek/eks/vllm-deepseek-32b-lws.yaml
    
### 조회하기 ###
```



```

### Pod 복구 테스트 하기 ###
```



```



## 마치며 ##
LWS는 "대형 LLM 분산 추론을 선언적으로 관리하고 싶다"라는 요구에서 출발해, 기존 StatefulSet + Service + Init Container 조합의 구성을 간결하게 바꿀 수 있다. 특히 vLLM 커뮤니티가 LWS를 표준 배포 방식으로 채택하고 있어서, 생태계 호환성 측면에서도 이점이 있다.


### 참고 - 파드 종료 흐름 ###

> [!TIP]
> 파드 종료 흐름 / kubelet이 파드를 죽이려 할 때 일어나는 일:
>
> 1. preStop 훅 실행 (있으면)
> 2. SIGTERM 전송 → "정리하고 종료해"
> 3. terminationGracePeriodSeconds 동안 대기
> 4. 아직 살아있으면 SIGKILL → "강제 종료"
> 기본값은 30초이나, 대부분의 실전 워크로드는 명시적으로 그 값을 늘려준다.
> 


#### preStop 훅 ####
graceful shutdown 제대로 하려면 preStop 훅을 추가:
```
lifecycle:
  preStop:
    exec:
      command:
        - /bin/sh
        - -c
        - |
          # 1. Readiness false로 만들어 LB에서 빠지게 함
          # 2. 진행 중 요청 완료 대기
          sleep 30
          # 3. 그 다음에야 SIGTERM이 앱에 감
```
```
흐름:

t=0   : preStop 실행 시작 (sleep 30)
t=0~30: LB가 이 파드를 endpoint에서 제거
        새 요청 안 들어옴, 기존 요청만 처리 중
t=30  : preStop 끝, SIGTERM 앱에 전달
t=30~120: 앱이 graceful shutdown (진행 요청 완료, cleanup)
t=120 : 아직 살아있으면 SIGKILL
grace period 안에 preStop 시간도 포함됩니다. 즉 preStop이 30초면, 실제 앱이 쓸 수 있는 시간은 120 - 30 = 90초.
```

## 레퍼런스 ##
* [vllm lws](https://docs.vllm.ai/en/latest/deployment/frameworks/lws/)
* [k8s volume](https://kubernetes.io/docs/concepts/storage/volumes/)
