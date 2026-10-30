# 실행 방법

## 요구 환경

- Python 3.11 이상
- 저장소 루트에서 실행

## 설치

PowerShell에서 다음 명령으로 필요한 패키지를 설치한다.

```powershell
python -m pip install -r requirements.txt
```

## 분석 실행

다음 한 명령으로 데이터 품질 검사, lot 결합, 영향 범위 집계와 보고서용 그림을 생성한다.

```powershell
python analyze.py
```

입력과 출력 폴더를 바꾸려면 다음처럼 실행한다.

```powershell
python analyze.py --data-dir data --output-dir output --customer-output AFFECTED_LOTS_고객A.csv
```

## 입력

분석은 `data/`의 다음 원본 CSV를 읽으며 수정하지 않는다.

- `lot_history.csv`
- `metrology_gox.csv`
- `eds_summary.csv`
- `shipments.csv`

설비·교정·FA·고객 문의 TXT는 보고서 작성 근거이며 Phase 1 계산 입력에는 사용하지 않는다.

## 출력

- `output/data_quality.csv`: 데이터 품질 검사 결과
- `output/metrology_corrected.csv`: 원측정값과 MT-02 잠정 보정값
- `output/lot_summary.csv`: 64개 lot의 공정·계측·EDS·출하 결합 결과
- `output/affected_lots_internal.csv`: 공식 공정 기준의 전체 영향 lot 8개
- `output/df03_boundary_lots.csv`: 마지막 정상 및 교체 후 첫 정상 DF-03 lot
- `output/core_summary.json`: 핵심 집계와 L6035 검산값
- `output/figures/gox_eds_by_lot.png`: lot별 GOX·EDS 추이
- `output/figures/gox_vs_hot_refresh.png`: GOX와 EDS 고온 불량률 관계
- `AFFECTED_LOTS_고객A.csv`: 고객A 전달용 영향 lot 5개. UTF-8 BOM 없이 `LOT_ID,SHIP_DATE,MODULE_QTY` 순서로 생성하며 합계 행은 넣지 않는다.

## 실패 시 확인

필수 열, 중복 키, lot별 공정 순서, lot당 측정 수, 입력 파일 간 lot 집합 또는 확정 영향 lot이 예상과 다르면 분석은 오류로 종료한다. 오류 메시지에서 파일과 검사 항목을 확인한 뒤 원자료 담당자에게 확인하며, 분석 코드가 원자료를 임의로 보정하거나 제외하지 않는다.

## 테스트

자동 테스트는 Phase 5에서 구현한다. 구현 후 실행 명령은 다음과 같이 사용한다.

```powershell
python -m pytest
```
