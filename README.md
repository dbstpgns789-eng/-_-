# 전산통계학 과제

전산통계학 수업 과제 모음입니다.

## 📚 과제 목록

### HW1 - 파레토 차트 (Pareto Chart)
- **파일**: `hw1_윤세훈 (6).ipynb`
- **주요 개념**:
  - `pd.cut()`: 연속형 데이터를 범주형으로 변환
  - `pd.crosstab()`: 교차 빈도표 생성
  - `.cumsum()`: 누적 합계 계산
  - `twinx()`: 이중 Y축 그래프
  - 파레토 차트: 빈도 막대 + 누적% 선 그래프

### HW2 - 큰 수의 법칙 (Law of Large Numbers)
- **파일**: `hw2_윤세훈 (2).ipynb`
- **주요 개념**:
  - `np.random.choice()`: 랜덤 샘플링
  - `np.cumsum()`: 누적 합계
  - 큰 수의 법칙: 시행 횟수 증가 시 이론적 확률로 수렴
  - 누적 평균: 각 시점까지의 평균 확률 계산
  - `plt.axhline()`: 기준선 표시

### HW3 - 모비율 신뢰구간 (Confidence Interval for Proportion)
- **파일**: `hw3_윤세훈 (2).ipynb`
- **주요 개념**:
  - `scipy.stats.norm`: 정규분포 객체
  - `norm.ppf()`: Percent Point Function (분위수 함수, 역CDF)
  - 점 추정량: 표본 비율 ($\hat{p}$)
  - 신뢰구간 공식: $\hat{p} \pm z_{\alpha/2} \times SE$
  - 표준오차: $\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$
  - 신뢰수준과 구간 폭의 Trade-off 관계

### HW4 - 허용 오차 내 최소 표본 크기 결정
- **파일**: `HW4_윤세훈 (2).ipynb`
- **문제**: 추정값이 실제 값과 ±0.27 이내 차이 보장 (90% 신뢰수준) → 최소 표본 수는?
- **주요 개념**:
  - 표본 크기 역산 공식: $n = \left(\frac{z \times \sigma}{E}\right)^2$
  - `math.ceil()`: 올림 함수 (조건 만족 최소값 보장)
  - 허용 오차(E): 추정치가 참값과 차이나도 되는 최대 범위
  - 로직: 신뢰구간 공식에서 n을 역산 → 소수점 올림
  - Trade-off: 정밀도↑/신뢰도↑ → 표본 크기↑

### HW5 - 다중선형회귀 (Multiple Linear Regression)
- **파일**: `HW5 (1).ipynb`
- **문제**: Boston 데이터에서 TAX와 가장 관련 높은 변수 4개로 회귀모형 구축
- **주요 로직**:
  1. **피어슨 상관계수**: 모든 변수와 TAX 간 상관도 계산 → |r| 상위 4개 선택
  2. **정규방정식**: $\hat{\beta} = (X^T X)^{-1} X^T y$ (최소제곱법 해석해)
  3. **t-검정**: 각 회귀계수 유의성 검정 (p-값 < 0.05)
  4. **R² / Adjusted R²**: 모델 설명력 평가 (변수 개수 보정)
- **핵심 개념**:
  - `np.linalg.inv()`: 역행렬 계산
  - SST(총제곱합), SSE(오차제곱합), R² = 1 - SSE/SST
  - Adjusted R²: 변수 증가 페널티 반영

### HW6
- **파일**: `HW6 (1).ipynb`

### HW7
- **파일**: `HW7 (2).ipynb`

## 🛠️ 사용 기술
- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## 📝 참고사항
- 각 과제는 독립적인 Jupyter Notebook으로 구성
- 통계 분석 및 데이터 시각화 중심
