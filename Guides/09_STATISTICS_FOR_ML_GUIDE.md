# Statistics for Machine Learning

Statistics provides the methods for reasoning about data when only a sample is available — quantifying uncertainty in estimates, comparing groups, testing whether observed differences are likely real, and detecting when patterns are chance findings. This guide covers applied statistics for machine learning practitioners: the framework of hypothesis testing, the most common statistical tests and the conditions under which each is appropriate, effect sizes, resampling-based inference, and the corrections required when many hypotheses are tested at once.

**Table of Contents**

1. [Descriptive vs Inferential Statistics](#1-descriptive-vs-inferential-statistics)
2. [Sampling and Sampling Distributions](#2-sampling-and-sampling-distributions)
3. [Confidence Intervals](#3-confidence-intervals)
4. [The Hypothesis Testing Framework](#4-the-hypothesis-testing-framework)
5. [Common Statistical Tests](#5-common-statistical-tests)
6. [Effect Sizes](#6-effect-sizes)
7. [Bootstrap and Permutation Tests](#7-bootstrap-and-permutation-tests)
8. [Multiple Comparison Corrections](#8-multiple-comparison-corrections)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Resources](#10-resources)

---

## 1. Descriptive vs Inferential Statistics

| Branch | Question | Examples |
|--------|----------|----------|
| Descriptive | What does this dataset look like? | Mean, median, standard deviation, histograms |
| Inferential | What does this sample suggest about the population? | Confidence intervals, hypothesis tests, regression coefficients |

Descriptive statistics summarize the data in hand; inferential statistics generalise from a sample to a population, with quantified uncertainty. Machine learning evaluation is fundamentally inferential — a held-out test set is a sample, and reported metrics are estimates with uncertainty.

```python
import numpy as np

x = np.array([2.1, 1.8, 3.3, 2.9, 2.0, 2.7, 3.1])

x.mean()              # central tendency
np.median(x)          # robust central tendency
x.std(ddof=1)         # sample standard deviation (ddof=1 for unbiased estimator)
np.percentile(x, [25, 75])   # interquartile range bounds
```

Note `ddof=1` for sample standard deviation — `ddof=0` (NumPy default) computes the population estimator and is biased downward for samples.

---

## 2. Sampling and Sampling Distributions

A sample is a subset drawn from a population. Statistics computed from samples — sample mean, sample variance — are themselves random variables. Their distribution across many hypothetical samples is the *sampling distribution* of the statistic.

The Central Limit Theorem (CLT) states that, for large enough sample sizes, the sampling distribution of the mean approaches a normal distribution regardless of the underlying population distribution. Most parametric inference rests on this result.

```python
import numpy as np

population = np.random.exponential(scale=1.0, size=100_000)   # heavily skewed

sample_means = [
    np.random.choice(population, size=100, replace=False).mean()
    for _ in range(10_000)
]
# np.array(sample_means) is approximately normal despite the skewed population
```

The standard error of the mean is `s / sqrt(n)` where `s` is the sample standard deviation and `n` the sample size. Standard error decreases with the square root of `n` — quadrupling the sample size halves the standard error.

---

## 3. Confidence Intervals

A confidence interval expresses uncertainty in a point estimate as a range. A 95% CI is constructed such that 95% of intervals built by the same procedure on different samples would contain the true population parameter.

```python
from scipy import stats
import numpy as np

x = np.random.normal(loc=10, scale=2, size=50)

ci_low, ci_high = stats.t.interval(
    confidence=0.95,
    df=len(x) - 1,
    loc=x.mean(),
    scale=stats.sem(x),
)
```

### 3.1 Common Misinterpretation

| Wrong | Correct |
|-------|---------|
| "There is a 95% probability the true mean is in this interval." | "If this procedure were repeated on many samples, 95% of the resulting intervals would contain the true mean." |

The probability statement applies to the procedure, not to any particular interval. A given interval either contains the true value or it does not.

### 3.2 Bootstrap Confidence Intervals

For statistics whose sampling distribution is unknown, bootstrap CIs are constructed empirically (Section 7).

---

## 4. The Hypothesis Testing Framework

Hypothesis testing is a structured procedure for deciding whether observed data are consistent with a default ("null") claim.

### 4.1 The Five Elements

| Element | Description |
|---------|-------------|
| Null hypothesis (H₀) | The default; typically "no effect" or "no difference" |
| Alternative hypothesis (H₁) | The claim being tested |
| Test statistic | A function of the data that summarizes evidence against H₀ |
| p-value | Probability of observing data at least as extreme as the observed, assuming H₀ is true |
| Significance level (α) | Pre-specified threshold (commonly 0.05); the maximum acceptable Type I error rate |

### 4.2 Decision Errors

| Decision | H₀ true | H₀ false |
|----------|---------|----------|
| Reject H₀ | Type I error (false positive), probability α | Correct (true positive), probability 1 − β |
| Fail to reject H₀ | Correct (true negative), probability 1 − α | Type II error (false negative), probability β |

Statistical *power* is 1 − β: the probability of correctly rejecting H₀ when it is false. Power increases with sample size, effect size, and α.

### 4.3 What a p-Value Is Not

A p-value is *not* the probability that H₀ is true. It is the probability of the observed data (or more extreme) under H₀. Common mis-statements:

| Wrong | Right |
|-------|-------|
| "p = 0.03 means there is a 3% chance the null is true." | "p = 0.03 means there is a 3% chance of seeing data this extreme if the null were true." |
| "p > 0.05 means H₀ is true." | "p > 0.05 means there is insufficient evidence to reject H₀." |

---

## 5. Common Statistical Tests

| Test | Use | Key assumption |
|------|-----|----------------|
| One-sample t-test | Compare sample mean to a known value | Approximately normal |
| Two-sample t-test (Welch's) | Compare means of two independent groups | Approximately normal; unequal variances allowed |
| Paired t-test | Compare matched pairs | Differences approximately normal |
| Wilcoxon signed-rank | Non-parametric paired alternative | Symmetric distribution of differences |
| Mann–Whitney U | Non-parametric two-sample alternative | Continuous distributions |
| One-way ANOVA | Compare means across 3+ groups | Normal; equal variance |
| Kruskal–Wallis | Non-parametric ANOVA | Continuous distributions |
| Chi-square test | Compare categorical distributions | Expected counts ≥ 5 in each cell |
| Fisher's exact test | Small-sample alternative to chi-square | None on sample size |
| Pearson correlation test | Linear association between numeric variables | Bivariate normal |
| Spearman correlation test | Monotonic association | None on distribution |

### 5.1 Two-Sample Comparison

```python
from scipy import stats

a = np.random.normal(loc=0, scale=1, size=30)
b = np.random.normal(loc=0.5, scale=1, size=30)

# Welch's t-test (unequal variances allowed)
t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

# Non-parametric alternative
u_stat, p_value = stats.mannwhitneyu(a, b)
```

### 5.2 Categorical Comparison

```python
import pandas as pd
from scipy import stats

# Observed contingency table
observed = pd.DataFrame(
    [[30, 20], [15, 35]],
    index=["GroupA", "GroupB"],
    columns=["Yes", "No"],
)

chi2, p_value, dof, expected = stats.chi2_contingency(observed)
```

### 5.3 Choosing a Test

The decision flow:

1. **Type of comparison?** Means, medians, proportions, association.
2. **One sample, two samples, or many?**
3. **Independent or paired observations?**
4. **Numeric or categorical outcome?**
5. **Are parametric assumptions reasonable?** If not, prefer the non-parametric alternative.

Non-parametric tests trade some power for fewer assumptions; with normal data and large samples, the difference in power is small.

---

## 6. Effect Sizes

A statistically significant result may be practically negligible if the effect is small but the sample is large. Effect sizes quantify the magnitude of an effect independent of sample size.

| Test | Effect-size measure |
|------|---------------------|
| Two-group mean comparison | Cohen's d |
| ANOVA | Eta-squared (η²), omega-squared (ω²) |
| Chi-square | Cramér's V, phi |
| Correlation | Pearson r itself; r² as variance explained |
| Regression coefficient | Standardized coefficient |

### 6.1 Cohen's d

```python
def cohens_d(a, b):
    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                         / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_std
```

Cohen's conventions for interpretation:

| Cohen's d | Conventional label |
|-----------|---------------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

These conventions are heuristics. Domain context determines what is *meaningful*; a "small" Cohen's d may be transformative in some contexts and irrelevant in others.

### 6.2 Reporting

Always report effect sizes alongside p-values. A 95% confidence interval on the effect size is more informative than either statistic alone.

---

## 7. Bootstrap and Permutation Tests

When parametric assumptions are uncomfortable or analytical solutions are unavailable, resampling produces confidence intervals and p-values empirically.

### 7.1 Bootstrap Confidence Interval

The bootstrap resamples the data with replacement many times and computes the statistic on each resample.

```python
import numpy as np

def bootstrap_ci(x, statistic=np.mean, n_iter=10_000, alpha=0.05):
    rng = np.random.default_rng(0)
    estimates = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(x, size=len(x), replace=True)
        estimates[i] = statistic(sample)
    return np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
```

The bootstrap is general — applicable to medians, regression coefficients, ROC AUC, or any statistic for which a closed-form standard error is unavailable.

### 7.2 Permutation Test

A permutation test compares observed group differences against the distribution of differences under random reassignment of group labels.

```python
def permutation_test(a, b, n_iter=10_000):
    observed = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(0)

    diffs = np.empty(n_iter)
    for i in range(n_iter):
        rng.shuffle(combined)
        diffs[i] = combined[:n_a].mean() - combined[n_a:].mean()
    p_value = (np.abs(diffs) >= np.abs(observed)).mean()
    return p_value
```

Permutation tests are exact under H₀ (no parametric distributional assumption) and easy to apply to non-standard statistics.

---

## 8. Multiple Comparison Corrections

Testing many hypotheses inflates the false-positive rate. With 20 independent tests at α = 0.05, the expected number of Type I errors is one — even if every null hypothesis is true.

| Method | Controls | Notes |
|--------|----------|-------|
| Bonferroni | Family-wise error rate (FWER) | Most conservative; multiply each p-value by `n_tests` |
| Holm–Bonferroni | FWER | Uniformly more powerful than Bonferroni |
| Benjamini–Hochberg | False discovery rate (FDR) | Less conservative; appropriate when many tests are conducted exploratorily |
| Sidak | FWER (assumes independence) | Slight gain over Bonferroni for independent tests |

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.008, 0.012, 0.041, 0.052, 0.110]

reject, p_adj, *_ = multipletests(p_values, alpha=0.05, method="fdr_bh")
```

Choosing between FWER and FDR control depends on the cost of a single false positive. For confirmatory analyses, FWER is appropriate; for exploratory screens (e.g., feature selection across many candidates), FDR is usually appropriate.

---

## 9. Common Pitfalls

| Pitfall | Mechanism | Mitigation |
|---------|-----------|------------|
| p-hacking | Trying many tests until one is significant | Pre-register analyses; correct for multiple testing |
| HARKing (Hypothesizing After Results Known) | Re-framing exploratory findings as confirmatory | Distinguish exploratory from confirmatory clearly |
| Confusing statistical and practical significance | Reporting p-values without effect sizes | Always report effect sizes and CIs |
| Ignoring assumptions | Applying parametric tests to non-normal data | Check assumptions; use non-parametric or robust alternatives |
| Selective reporting | Reporting only significant subgroups | Pre-specify subgroup analyses |
| Optional stopping | Continuing data collection until p < 0.05 | Pre-specify sample size; use sequential designs honestly |
| Treating "p > 0.05" as "no effect" | Absence of significance ≠ absence of effect | Report the effect size and CI; consider power |

The American Statistical Association's 2016 statement on p-values rejects mechanistic use ("p < 0.05 = significant") and recommends reporting in context with effect sizes, study design, and prior evidence.

---

## 10. Resources

- [Wasserstein and Lazar, *The ASA Statement on p-Values* (2016)](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108) — authoritative statement on appropriate use of p-values.
- [Cohen, *Statistical Power Analysis for the Behavioral Sciences* (1988)](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832) — foundational treatment of effect sizes and power.
- [Efron and Tibshirani, *An Introduction to the Bootstrap* (1993)](https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317) — comprehensive treatment of resampling methods.
- [Benjamini and Hochberg, *Controlling the False Discovery Rate* (1995)](https://www.jstor.org/stable/2346101) — original FDR paper.
- [`scipy.stats` documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) — every test and distribution covered above.
- [`statsmodels` documentation](https://www.statsmodels.org/) — regression, hypothesis tests, multiple-comparison correction.
- [Wilkinson and the Task Force on Statistical Inference, *Statistical Methods in Psychology Journals* (1999)](https://www.apa.org/pubs/journals/releases/amp-548594.pdf) — reporting guidelines applicable beyond psychology.

---

[← Previous: Mathematics for ML](08_MATH_FOR_ML_GUIDE.md) | [Index](README.md) | [Next: Datasets →](10_DATASETS_GUIDE.md)
