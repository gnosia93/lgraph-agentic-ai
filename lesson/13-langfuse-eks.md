## Langfuse 셀프호스팅: Docker에서 EKS까지 ##

### 1. 왜 셀프호스팅인가 ###

Langfuse Cloud로 빠르게 시작할 순 있지만, 트레이스에는 사용자 입력과 모델 출력이 그대로 담기므로, 민감한 데이터를 외부 SaaS로 전송할 수 없는 환경에서는 Langfuse를 자체 인프라에 직접 운영해야 한다. 데이터 거버넌스 요구, 사내 네트워크 격리, 비용 구조 등의 이유로 셀프호스팅을 선택한다.

이 편에서는 먼저 로컬에서 Docker Compose로 구조를 이해한 뒤, 같은 구성을 Amazon EKS에 배포해 팀이 함께 사용할 수 있는 인스턴스를 만든다.

### 2. 아키텍처 구성 요소 ###

Langfuse는 단일 프로세스가 아니라 여러 구성 요소로 이루어진다. 배포 전에 각 요소의 역할을 이해해야 운영 시 문제를 진단할 수 있다.

웹 서버(langfuse-web)는 대시보드 UI와 API를 제공한다. SDK가 보내는 트레이스 데이터를 받는 수집 엔드포인트도 여기에 포함된다. 워커(langfuse-worker)는 수집된 데이터를 비동기로 처리해 저장소에 적재한다. SDK가 데이터를 버퍼에 모아 전송하는 것과 짝을 이루는 서버 측 비동기 처리 단계다.

저장소는 세 가지로 나뉜다. PostgreSQL은 프로젝트, 사용자, 프롬프트, 설정 등 트랜잭션 데이터를 저장한다. ClickHouse는 트레이스와 Observation 같은 대량의 이벤트 데이터를 저장한다. 분석 질의에 최적화된 컬럼 기반 데이터베이스로, 트레이스 조회와 집계 성능을 담당한다. Redis는 큐와 캐시로 사용되어 웹 서버와 워커 사이의 작업 전달을 중계한다. 그리고 트레이스에 포함된 큰 페이로드(긴 입력·출력 등)는 S3 호환 오브젝트 스토리지에 저장한다.

요청 흐름은 다음과 같다. SDK가 웹 서버의 수집 엔드포인트로 데이터를 보내면, 웹 서버는 이를 Redis 큐에 넣는다. 워커가 큐에서 꺼내 PostgreSQL과 ClickHouse, 오브젝트 스토리지에 적재한다. 사용자가 대시보드를 조회하면 웹 서버가 ClickHouse와 PostgreSQL에서 데이터를 읽어 보여준다.

## 3. Docker Compose로 로컬 기동

먼저 로컬에서 전체 스택을 띄워 구조를 확인한다. 공식 저장소의 Compose 파일에는 위의 모든 구성 요소가 포함되어 있다.

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
```

기동이 완료되면 http://localhost:3000 에서 대시보드에 접속한다. 계정을 생성하고 프로젝트를 만든 뒤 API 키를 발급하면, 1편의 애플리케이션이 이 로컬 인스턴스로 트레이스를 보내도록 LANGFUSE_HOST를 http://localhost:3000 으로 지정해 연결할 수 있다.

이 단계의 목적은 운영이 아니라 구성 요소 간 관계를 직접 확인하는 것이다. docker compose ps로 웹, 워커, PostgreSQL, ClickHouse, Redis 컨테이너가 함께 떠 있는 것을 확인한다.

## 4. EKS 배포

운영 환경에서는 각 구성 요소를 쿠버네티스 워크로드로 배포한다. Langfuse는 공식 Helm 차트를 제공하므로 이를 사용한다.

### 4.1 사전 준비

EKS 클러스터와 kubectl, helm이 필요하다. 저장소는 두 가지 방식으로 운영할 수 있다. 하나는 PostgreSQL, ClickHouse, Redis, 오브젝트 스토리지를 모두 클러스터 안에 함께 배포하는 방식이고, 다른 하나는 관리형 서비스를 사용하는 방식이다. 운영 환경에서는 후자를 권장한다. PostgreSQL은 Amazon RDS, 오브젝트 스토리지는 Amazon S3, 캐시는 Amazon ElastiCache(Redis)로 분리하면 상태 저장소를 클러스터 수명과 독립적으로 관리할 수 있고 백업과 확장이 용이하다. ClickHouse는 관리형 선택지가 제한적이므로 클러스터 내 배포 또는 별도 운영을 검토한다.

### 4.2 시크릿 준비

배포 전에 민감 값을 시크릿으로 등록한다. Langfuse는 데이터 암호화와 세션에 사용하는 키를 요구한다.

```bash
kubectl create namespace langfuse

kubectl -n langfuse create secret generic langfuse-secrets \
  --from-literal=salt="$(openssl rand -hex 16)" \
  --from-literal=encryption-key="$(openssl rand -hex 32)" \
  --from-literal=nextauth-secret="$(openssl rand -hex 32)"
```

encryption-key는 트레이스 데이터 암호화에 사용되므로 분실하면 기존 데이터를 복호화할 수 없다. 안전하게 보관한다. 실제 운영에서는 이 값들을 AWS Secrets Manager나 External Secrets Operator로 관리하는 것이 바람직하다.

### 4.3 Helm 설치

차트 저장소를 추가하고 values 파일로 구성을 지정해 설치한다.

```bash
helm repo add langfuse https://langfuse.github.io/langfuse-k8s
helm repo update
```

values.yaml 예시는 다음과 같다. 관리형 저장소를 사용하는 구성의 골격이다.

```yaml
langfuse:
  salt:
    secretKeyRef:
      name: langfuse-secrets
      key: salt
  encryptionKey:
    secretKeyRef:
      name: langfuse-secrets
      key: encryption-key
  nextauth:
    secret:
      secretKeyRef:
        name: langfuse-secrets
        key: nextauth-secret

# 관리형 PostgreSQL(RDS) 사용
postgresql:
  deploy: false
  host: <rds-endpoint>
  auth:
    username: langfuse
    database: langfuse

# 관리형 Redis(ElastiCache) 사용
redis:
  deploy: false
  host: <elasticache-endpoint>

# S3 오브젝트 스토리지
s3:
  deploy: false
  bucket: <bucket-name>
  region: <aws-region>

# ClickHouse는 클러스터 내 배포
clickhouse:
  deploy: true
```

```bash
helm install langfuse langfuse/langfuse \
  -n langfuse -f values.yaml
```

설치 후 파드 상태를 확인한다.

```bash
kubectl -n langfuse get pods
```

웹과 워커, ClickHouse 파드가 Running 상태가 되면 배포가 완료된 것이다. 관리형 저장소를 쓰지 않고 클러스터 내 배포를 선택했다면 PostgreSQL과 Redis 파드도 함께 보인다.

### 4.4 노출과 접근 제어

대시보드를 외부에 노출하려면 인그레스를 설정한다. EKS에서는 AWS Load Balancer Controller를 통해 ALB로 노출하는 것이 일반적이다. 인그레스에는 반드시 인증과 TLS를 적용한다. 트레이스에 민감 데이터가 포함되므로, 인증 없이 노출된 엔드포인트는 데이터 유출 경로가 된다. 사내에서만 사용한다면 내부 ALB나 프라이빗 서브넷으로 한정하고, 외부 접근이 필요하면 OIDC 연동이나 WAF를 함께 구성한다.

## 5. 운영 시 고려 사항

워커는 수집량에 따라 스케일링한다. 트래픽이 늘면 SDK가 보내는 데이터가 Redis 큐에 쌓이므로, 큐 적체가 관측되면 워커 레플리카를 늘린다. 웹 서버도 조회 부하에 따라 수평 확장한다.

데이터 보존 정책을 정한다. ClickHouse에 쌓이는 트레이스는 시간이 지나면 용량이 커지므로, 보존 기간을 정해 오래된 데이터를 삭제하거나 아카이브한다. 규제 요구에 따라 PII 마스킹이나 보존 한도를 적용한다.

백업은 PostgreSQL과 ClickHouse, 오브젝트 스토리지 각각에 대해 수행한다. 관리형 서비스를 사용하면 RDS 자동 백업과 S3 버전 관리로 상당 부분을 위임할 수 있다. encryption-key를 포함한 시크릿도 별도로 안전하게 백업한다.

모니터링은 Langfuse 자체의 파드 상태, Redis 큐 길이, ClickHouse 질의 지연을 관측 대상으로 둔다. 큐가 계속 증가하면 워커 처리량이 수집량을 따라가지 못하는 상태이므로 스케일 조정이 필요하다.


