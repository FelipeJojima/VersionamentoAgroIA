import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import AnovaRM
from baycomp import SignedRankTest
from collections import namedtuple

__all__ = ['rank_two', 'rank_multiple_normal_homoscedastic', 'rank_bayesian', 'RankResult',
           'rank_multiple_nonparametric', 'cd_diagram', 'get_sorted_rank_groups', 'ci_plot', 'test_normality', 'posterior_maps']


class RankResult(namedtuple('RankResult', ('rankdf', 'pvalue', 'cd', 'omnibus', 'posthoc', 'all_normal',
                                           'pvals_shapiro', 'homoscedastic', 'pval_homogeneity', 'homogeneity_test',
                                           'alpha', 'alpha_normality', 'num_samples', 'sample_matrix',
                                           'posterior_matrix', 'decision_matrix', 'rope', 'rope_mode', 'effect_size',
                                           'force_mode', 'plot_order'))):
    __slots__ = ()

    def __str__(self):
        return 'RankResult(rankdf=\n%s\n' \
               'pvalue=%s\n' \
               'cd=%s\n' \
               'omnibus=%s\n' \
               'posthoc=%s\n' \
               'all_normal=%s\n' \
               'pvals_shapiro=%s\n' \
               'homoscedastic=%s\n' \
               'pval_homogeneity=%s\n' \
               'homogeneity_test=%s\n' \
               'alpha=%s\n' \
               'alpha_normality=%s\n' \
               'num_samples=%s\n' \
               'posterior_matrix=\n%s\n' \
               'decision_matrix=\n%s\n' \
               'rope=%s\n' \
               'rope_mode=%s\n' \
               'effect_size=%s\n'\
               'force_mode=%s)' % (self.rankdf, self.pvalue, self.cd, self.omnibus, self.posthoc, self.all_normal,
                                self.pvals_shapiro, self.homoscedastic, self.pval_homogeneity,
                                self.homogeneity_test, self.alpha, self.alpha_normality, self.num_samples,
                                self.posterior_matrix, self.decision_matrix, self.rope, self.rope_mode,
                                self.effect_size, self.force_mode)


class _ComparisonResult(namedtuple('ComparisonResult', ('rankdf', 'pvalue', 'cd', 'omnibus', 'posthoc',
                                                        'effect_size', 'reorder_pos'))):
    __slots__ = ()

    def __str__(self):
        return '_ComparisonResult(rankdf=\n%s\n' \
               'pvalue=%s\n' \
               'cd=%s\n' \
               'omnibus=%s\n' \
               'posthoc=%s\n' \
               'effect_size=%s\n' \
               'reorder_pos=%s)' % (self.rankdf, self.pvalue, self.cd, self.omnibus, self.posthoc, self.effect_size,
                                    self.reorder_pos)


class _BayesResult(namedtuple('BayesResult', ('rankdf', 'sample_matrix', 'posterior_matrix', 'decision_matrix', 'effect_size',
                                              'reorder_pos'))):
    __slots__ = ()

    def __str__(self):
        return 'BayesResult(rankdf=\n%s\n' \
               'posterior_matrix=%s\n' \
               'decision_matrix=%s\n' \
               'effect_size=%s\n' \
               'reorder_pos=%s)' % (self.rankdf, self.posterior_matrix, self.decision_matrix, self.effect_size,
                                    self.reorder_pos)


def _pooled_std(x, y):
    """
    Calculate the pooled standard deviation of x and y
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


def _pooled_mad(x, y):
    """
    Calculate the pooled median absolute deviation of x and y
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    mad_x = stats.median_abs_deviation(x, scale=1/1.4826)  # scale MAD to be similar to SD of a normal
    mad_y = stats.median_abs_deviation(y, scale=1/1.4826)  # scale MAD to be similar to SD of a normal
    return np.sqrt(((nx - 1) * mad_x ** 2 + (ny - 1) * mad_y ** 2) / dof)


def _cohen_d(x, y):
    """
    Calculate the effect size using Cohen's d
    """
    return (np.mean(x) - np.mean(y)) / _pooled_std(x, y)


def _akinshin_gamma(x, y):
    """
    Calculate the effect size using a non-parametric variant of Cohen's d that replaces the pooled
    standard deviation with the pooled median absolute deviation. This metric is based on this blog
    post (no publication yet).
    https://aakinshin.net/posts/nonparametric-effect-size/
    """
    return (np.median(x) - np.median(y)) / _pooled_mad(x, y)


def _cliffs_delta(x, y):
    """
    Calculates Cliff's delta.
    """
    delta = 0
    for x_val in x:
        result = 0
        for y_val in y:
            if y_val > x_val:
                result -= 1
            elif x_val > y_val:
                result += 1
        delta += result / len(y)
    if abs(delta) < 10e-16:
        # due to minor rounding errors
        delta = 0
    else:
        delta = delta / len(x)
    return delta


def _effect_level(effect_size, method='cohen_d'):
    """
    Determines magnitude of effect size.
    """
    if not isinstance(method, str):
        raise TypeError('method must be of type str')
    if method not in ['cohen_d', 'cliff_delta', 'akinshin_gamma']:
        raise ValueError("method must be one of the following strings: 'cohen_d', 'cliff_delta', 'akinshin_gamma'")
    effect_size = abs(effect_size)
    if method == 'cliff_delta':
        if effect_size < 0.147:
            return 'negligible'
        elif effect_size < 0.33:
            return 'small'
        elif effect_size < 0.474:
            return 'medium'
        else:
            return 'large'
    if method == 'cohen_d' or method == 'akinshin_gamma':
        if effect_size < 0.2:
            return 'negligible'
        elif effect_size < 0.5:
            return 'small'
        elif effect_size < 0.8:
            return 'medium'
        else:
            return 'large'


def _critical_distance(alpha, k, n):
    """
    Determines the critical distance for the Nemenyi test with infinite degrees of freedom.
    """
    return qsturng(1 - alpha, k, np.inf) * np.sqrt(k * (k + 1) / (12 * n))


def _confidence_interval(data, alpha, is_normal=True):
    """
    Determines the confidence interval.
    """
    if is_normal:
        mean = data.mean()
        ci_range = data.sem() * stats.t.ppf((1 + 1 - alpha) / 2, len(data) - 1)
        return mean - ci_range, mean + ci_range
    else:
        quantile = stats.norm.ppf(1 - (alpha / 2))
        r = (len(data) / 2) - (quantile * np.sqrt(len(data) / 2))
        s = 1 + (len(data) / 2) + (quantile * np.sqrt(len(data) / 2))

        r = max(0, r)
        s = min(len(data)-1, s)
        sorted_data = data.sort_values()
        lower = sorted_data.iloc[int(round(r))]
        upper = sorted_data.iloc[int(round(s))]
        return lower, upper


def _posterior_decision(probabilities, alpha):
    """
    calculate decision based on probabilities and desired significance
    """
    if len(probabilities) == 3:
        # with ROPE
        if probabilities[0] >= 1 - alpha:
            return 'smaller'
        elif probabilities[1] >= 1 - alpha:
            return 'equal'
        elif probabilities[2] >= 1 - alpha:
            return 'larger'
        else:
            return 'inconclusive'
    else:
        # without ROPE (i.e., rope=0)
        if probabilities[0] >= 1 - alpha:
            return 'smaller'
        elif probabilities[1] >= 1 - alpha:
            return 'larger'
        else:
            return 'inconclusive'


def rank_two(data, alpha, verbose, all_normal, order, effect_size, force_mode):
    """
    Uses paired t-test for normal data and Wilcoxon's signed rank test for other distributions. Can be overridden with
    force_mode.
    """
    larger = np.argmax(data.median().values)
    smaller = int(bool(larger - 1))
    if (force_mode is not None and force_mode == 'parametric') or (force_mode is None and all_normal):
        if verbose:
            print("Using paired t-test")
        omnibus = 'ttest'
        pval = stats.ttest_rel(data.iloc[:, larger], data.iloc[:, smaller]).pvalue
    else:
        if verbose:
            print("Using Wilcoxon's signed rank test (one-sided)")
        omnibus = 'wilcoxon'
        pval = stats.wilcoxon(data.iloc[:, larger], data.iloc[:, smaller], alternative='greater').pvalue
    if verbose:
        if pval >= alpha:
            print(
                "Fail to reject null hypothesis that there is no difference between the distributions (p=%f)" % pval)
        else:
            print("Rejecting null hypothesis that there is no difference between the distributions (p=%f)" % pval)
    rankdf, effsize_method, reorder_pos = _create_result_df_skeleton(data, alpha, all_normal, order,
                                                                     effect_size=effect_size, force_mode=force_mode)
    return _ComparisonResult(rankdf, pval, None, omnibus, None, effsize_method, reorder_pos)


def rank_multiple_normal_homoscedastic(data, alpha, verbose, order, effect_size, force_mode):
    """
    Analyzes data using repeated measures ANOVA and Tukey HSD.
    """
    stacked_data = data.stack().reset_index()
    stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                'level_1': 'treatment',
                                                0: 'result'})
    anova = AnovaRM(stacked_data, 'result', 'id', within=['treatment'])
    pval = anova.fit().anova_table['Pr > F'].iat[0]
    if verbose:
        if pval >= alpha:
            print(
                "Fail to reject null hypothesis that there is no difference between the distributions (p=%f)" % pval)
        else:
            print("Rejecting null hypothesis that there is no difference between the distributions (p=%f)" % pval)
            print(
                "Using Tukey HSD post hoc test.",
                "Differences are significant if the confidence intervals of the mean values are not overlapping.")

    multicomp = MultiComparison(stacked_data['result'], stacked_data['treatment'])
    tukey_res = multicomp.tukeyhsd()
    # must create plot to get confidence intervals
    tukey_res.plot_simultaneous()
    # delete plot instead of showing
    plt.close()

    rankdf, effsize_method, reorder_pos = _create_result_df_skeleton(data, None, True, order,
                                                                     effect_size=effect_size, force_mode=force_mode)
    for population in rankdf.index:
        mean = data.loc[:, population].mean()
        ci_range = tukey_res.halfwidths[data.columns.get_loc(population)]
        lower, upper = mean - ci_range, mean + ci_range
        rankdf.at[population, 'ci_lower'] = lower
        rankdf.at[population, 'ci_upper'] = upper
    return _ComparisonResult(rankdf, pval, None, 'anova', 'tukeyhsd', effsize_method, reorder_pos)


def rank_multiple_nonparametric(data, alpha, verbose, all_normal, order, effect_size, force_mode):
    """
    Analyzes data following Demsar using Friedman-Nemenyi.
    """
    if verbose:
        print("Using Friedman test as omnibus test")
    pval = stats.friedmanchisquare(*data.transpose().values).pvalue
    if verbose:
        if pval >= alpha:
            print("Fail to reject null hypothesis that there is no difference between the distributions (p=%f)" % pval)
        else:
            print("Rejecting null hypothesis that there is no difference between the distributions (p=%f)" % pval)
            print(
                "Using Nemenyi post-hoc test.",
                "Differences are significant,"
                "if the distance between the mean ranks is greater than the critical distance.")
    cd = _critical_distance(alpha, k=len(data.columns), n=len(data))
    rankdf, effsize_method, reorder_pos = _create_result_df_skeleton(data, alpha, all_normal, order,
                                                                     effect_size=effect_size, force_mode=force_mode)
    return _ComparisonResult(rankdf, pval, cd, 'friedman', 'nemenyi', effsize_method, reorder_pos)


def rank_bayesian(data, alpha, verbose, all_normal, order, rope, rope_mode, nsamples, effect_size, random_state, force_mode):
    # TODO check if some outputs for the verbose mode would be helpful
    if (force_mode is not None and force_mode == 'parametric') or (force_mode is None and all_normal):
        order_column = 'mean'
    else: # either with force_mode or not all_normal
        order_column = 'median'
    result_df, effsize_method, reorder_pos = _create_result_df_skeleton(data, alpha/len(data.columns), all_normal,
                                                                        order, order_column, effect_size, force_mode)
    result_df = result_df.drop('meanrank', axis='columns')
    result_df['p_equal'] = np.nan
    result_df['p_smaller'] = np.nan
    result_df['decision'] = 'NA'
    result_df['p_equal_above'] = np.nan
    result_df['p_smaller_above'] = np.nan


    # re-order columns to have the same order as results
    reordered_data = data.reindex(result_df.index, axis=1)

    sample_matrix = pd.DataFrame(index=reordered_data.columns, columns=reordered_data.columns)
    posterior_matrix = pd.DataFrame(index=reordered_data.columns, columns=reordered_data.columns)
    decision_matrix = pd.DataFrame(index=reordered_data.columns, columns=reordered_data.columns)
    for i in range(len(data.columns)):
        for j in range(i+1, len(reordered_data.columns)):
            if rope_mode == 'effsize':
                # half the size of a small effect size following Kruschke (2018)
                if (force_mode is not None and force_mode == 'parametric') or (force_mode is None and all_normal):
                    # use Cohen's d
                    cur_rope = rope*_pooled_std(reordered_data.iloc[:, i], reordered_data.iloc[:, j])
                else: # either with force_mode or not all_normal
                    # use Akinshin's gamma
                    cur_rope = rope*_pooled_mad(reordered_data.iloc[:, i], reordered_data.iloc[:, j])
            elif rope_mode == 'absolute':
                cur_rope = rope
            else:
                raise ValueError("Unknown rope_mode method, this should not be possible.")
            sample = SignedRankTest(x=reordered_data.iloc[:, i], y=reordered_data.iloc[:, j], rope=cur_rope,
                                    nsamples=nsamples, random_state=random_state)
            posterior_probabilities = sample.probs()
            sample_matrix.iloc[i, j] = sample
            posterior_matrix.iloc[i, j] = posterior_probabilities
            decision_matrix.iloc[i, j] = _posterior_decision(posterior_probabilities, alpha)
            decision_matrix.iloc[j, i] = _posterior_decision(posterior_probabilities[::-1], alpha)
            if i == 0:
                # comparison with "best"
                result_df.loc[result_df.index[j], 'p_equal'] = posterior_probabilities[1]
                result_df.loc[result_df.index[j], 'p_smaller'] = posterior_probabilities[0]
                result_df.loc[result_df.index[j], 'decision'] = _posterior_decision(posterior_probabilities, alpha)
    for i in range(1, len(data.columns)):
        result_df.loc[result_df.index[i], 'p_equal_above'] = posterior_matrix.iloc[i-1, i][1]
        result_df.loc[result_df.index[i], 'p_smaller_above'] = posterior_matrix.iloc[i-1, i][0]
        result_df.loc[result_df.index[i], 'decision_above'] = _posterior_decision(posterior_matrix.iloc[i-1, i], alpha)

    return _BayesResult(result_df, sample_matrix, posterior_matrix, decision_matrix, effsize_method, reorder_pos)


def _create_result_df_skeleton(data, alpha, all_normal, order, order_column='meanrank', effect_size=None,
                               force_mode=None):
    """
    Creates data frame for results. CI may be left empty in case alpha is None
    """
    if effect_size is None:
        if all_normal:
            effsize_method = 'cohen_d'
        else:
            effsize_method = 'akinshin_gamma'
    else:
        effsize_method = effect_size

    asc = None
    if order == 'descending':
        asc = False
    elif order == 'ascending':
        asc = True

    rankmat = data.rank(axis='columns', ascending=asc)
    meanranks = rankmat.mean()
    if (force_mode is not None and force_mode=='parametric') or (force_mode is None and all_normal):
        rankdf = pd.DataFrame(index=meanranks.index,
                              columns=['meanrank', 'mean', 'std', 'ci_lower', 'ci_upper', 'effect_size', 'magnitude', 'effect_size_above', 'magnitude_above'])
        rankdf['mean'] = data.mean().reindex(meanranks.index)
        rankdf['std'] = data.std().reindex(meanranks.index)
    else:
        rankdf = pd.DataFrame(index=meanranks.index,
                              columns=['meanrank', 'median', 'mad', 'ci_lower', 'ci_upper', 'effect_size', 'magnitude', 'effect_size_above', 'magnitude_above'])
        rankdf['median'] = data.median().reindex(meanranks.index)
        for population in rankdf.index:
            rankdf.at[population, 'mad'] = stats.median_abs_deviation(data.loc[:, population])
    rankdf['meanrank'] = meanranks

    # need to know reordering here (see issue #7)
    reorder_index = rankdf[order_column].sort_values(ascending=asc).index
    reorder_pos = [reorder_index.get_loc(old_index) for old_index in rankdf.index]
    rankdf = rankdf.reindex(reorder_index)

    population_above = None
    for population in rankdf.index:
        if effsize_method == 'cohen_d':
            effsize = _cohen_d(data.loc[:, rankdf.index[0]], data.loc[:, population])
            if population_above is not None:
                effsize_above = _cohen_d(data.loc[:, population_above], data.loc[:, population])
        elif effsize_method == 'cliff_delta':
            effsize = _cliffs_delta(data.loc[:, rankdf.index[0]], data.loc[:, population])
            if population_above is not None:
                effsize_above = _cliffs_delta(data.loc[:, population_above], data.loc[:, population])
        elif effsize_method == 'akinshin_gamma':
            effsize = _akinshin_gamma(data.loc[:, rankdf.index[0]], data.loc[:, population])
            if population_above is not None:
                effsize_above = _akinshin_gamma(data.loc[:, population_above], data.loc[:, population])
        else:
            raise ValueError("Unknown effsize method, this should not be possible.")
        if population_above is None:
            effsize_above = 0.0
        rankdf.at[population, 'effect_size'] = effsize
        rankdf.at[population, 'magnitude'] = _effect_level(effsize, effsize_method)
        rankdf.at[population, 'effect_size_above'] = effsize_above
        rankdf.at[population, 'magnitude_above'] = _effect_level(effsize_above, effsize_method)

        if alpha is not None:
            lower, upper = _confidence_interval(data.loc[:, population], alpha / len(data.columns),
                                                is_normal=all_normal)
            rankdf.at[population, 'ci_lower'] = lower
            rankdf.at[population, 'ci_upper'] = upper
        population_above = population

    return rankdf, effsize_method, reorder_pos


def get_sorted_rank_groups(result, reverse):
    if reverse:
        names = result.rankdf.iloc[::-1].index.to_list()
        if result.cd is not None:
            sorted_ranks = result.rankdf.iloc[::-1].meanrank
            critical_difference = result.cd
        else:
            sorted_ranks = result.rankdf.iloc[::-1]['mean']
            critical_difference = (result.rankdf.ci_upper[0] - result.rankdf.ci_lower[0]) / 2
    else:
        names = result.rankdf.index.to_list()
        if result.cd is not None:
            sorted_ranks = result.rankdf.meanrank
            critical_difference = result.cd
        else:
            sorted_ranks = result.rankdf['mean']
            critical_difference = (result.rankdf.ci_upper.iloc[0] - result.rankdf.ci_lower.iloc[0]) / 2

    groups = []
    cur_max_j = -1
    for i in range(len(sorted_ranks)):
        max_j = None
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks.iloc[i] - sorted_ranks.iloc[j]) <= critical_difference:
                max_j = j
                # print(i, j)
        if max_j is not None and max_j > cur_max_j:
            cur_max_j = max_j
            groups.append((i, max_j))
    return sorted_ranks, names, groups


def cd_diagram(result, reverse, ax, width):
    """
    Creates a Critical Distance diagram.
    """

    def plot_line(line, color='k', **kwargs):
        ax.plot([pos[0] / width for pos in line], [pos[1] / height for pos in line], color=color, **kwargs)

    def plot_text(x, y, s, *args, **kwargs):
        ax.text(x / width, y / height, s, *args, **kwargs)

    result_copy = RankResult(**result._asdict())
    result_copy = result_copy._replace(rankdf=result.rankdf.sort_values(by='meanrank'))
    sorted_ranks, names, groups = get_sorted_rank_groups(result_copy, reverse)
    cd = result.cd

    lowv = min(1, int(math.floor(min(sorted_ranks))))
    highv = max(len(sorted_ranks), int(math.ceil(max(sorted_ranks))))
    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            relative_rank = rank - lowv
        else:
            relative_rank = highv - rank
        return textspace + scalewidth / (highv - lowv) * relative_rank

    linesblank = 0.2 + 0.2 + (len(groups) - 1) * 0.1

    # add scale
    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((len(sorted_ranks) + 1) / 2) * 0.2 + minnotsignificant

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    plot_line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2),
                   (rankpos(a), cline)],
                  linewidth=0.7)

    for a in range(lowv, highv + 1):
        plot_text(rankpos(a), cline - tick / 2 - 0.05, str(a),
                  ha="center", va="bottom")

    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line([(rankpos(sorted_ranks.iloc[i]), cline),
                   (rankpos(sorted_ranks.iloc[i]), chei),
                   (textspace - 0.1, chei)],
                  linewidth=0.7)
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line([(rankpos(sorted_ranks.iloc[i]), cline),
                   (rankpos(sorted_ranks.iloc[i]), chei),
                   (textspace + scalewidth + 0.1, chei)],
                  linewidth=0.7)
        plot_text(textspace + scalewidth + 0.2, chei, names[i],
                  ha="left", va="center")

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv + cd)
    else:
        begin, end = rankpos(highv), rankpos(highv - cd)

    plot_line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
    plot_line([(begin, distanceh + bigtick / 2),
               (begin, distanceh - bigtick / 2)],
              linewidth=0.7)
    plot_line([(end, distanceh + bigtick / 2),
               (end, distanceh - bigtick / 2)],
              linewidth=0.7)
    plot_text((begin + end) / 2, distanceh - 0.05, "CD",
              ha="center", va="bottom")

    # no-significance lines
    side = 0.05
    no_sig_height = 0.1
    start = cline + 0.2
    for l, r in groups:
        plot_line([(rankpos(sorted_ranks.iloc[l]) - side, start),
                   (rankpos(sorted_ranks.iloc[r]) + side, start)],
                  linewidth=2.5)
        start += no_sig_height

    return ax


def ci_plot(result, reverse, ax, width):
    """
    Uses error bars to create a plot of the confidence intervals of the mean value.
    """
    if result.plot_order is not None:
        # if the result has a plot order, use it
        ordered_df = result.rankdf.loc[result.plot_order]
    else:
        # otherwise, use the default order
        ordered_df = result.rankdf

    # we usually revert, because the plot has the first list item at the bottom
    if reverse:
        sorted_df = ordered_df.iloc[::-1]
    else:
        sorted_df = ordered_df
    sorted_means = sorted_df['mean']
    ci_lower = sorted_df.ci_lower
    ci_upper = sorted_df.ci_upper
    names = sorted_df.index

    height = len(sorted_df)
    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor('white')
        ax = plt.gca()
    ax.errorbar(sorted_means, range(len(sorted_means)), xerr=(ci_upper.iloc[0] - ci_lower.iloc[0]) / 2, marker='o',
                linestyle='None', color='k', ecolor='k')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names.to_list())
    ax.set_title('%.1f%% Confidence Intervals of the Mean' % ((1 - result.alpha) * 100))
    return ax


def test_normality(data, alpha, verbose):
    """
    Tests if all populations are normal and return whether this is true and a list of p-values
    """
    all_normal = True
    pvals_shapiro = []
    for column in data.columns:
        w, pval_shapiro = stats.shapiro(data[column])
        pvals_shapiro.append(pval_shapiro)
        if pval_shapiro < alpha:
            all_normal = False
            if verbose:
                print("Rejecting null hypothesis that data is normal for column %s (p=%f<%f)" % (
                    column, pval_shapiro, alpha))
        elif verbose:
            print("Fail to reject null hypothesis that data is normal for column %s (p=%f>=%f)" % (
                column, pval_shapiro, alpha))
    return all_normal, pvals_shapiro


def _heatmap(data, row_labels, col_labels, ax,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Adopted from the scikit-learn documentation:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-')#, linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def _annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     vrange=None, **textkw):
    """
    A function to annotate a heatmap.

    Adopted from the scikit-learn documentation:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if vrange is not None:
        threshold = (vrange[1]-vrange[0])/2
    else:
        threshold = im.norm(data.max())/2

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    if textcolors is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                im.axes.text(j, i, valfmt(data[i, j], None), **kw)


def _create_annotated_heatmap(ax, data, title_prefix, cmap, annot_color, cbarlabel, cbar_kw=None, vmin=0, vmax=1):
    """
    Creates an annotated heatmap on the given axes.
    """
    im = _heatmap(data.values, data.index, data.columns,
                        ax=ax, cbarlabel=cbarlabel,
                        cmap=cmap, vmin=vmin, vmax=vmax, cbar_kw=cbar_kw)
    _annotate_heatmap(im, valfmt="{x:.2f}", vrange=(vmin, vmax), textcolors=annot_color)
    ax.set_title(title_prefix + cbarlabel)
    ax.set_xlabel("B")
    ax.set_ylabel("A", rotation=0)

def posterior_maps(result, *, width, cmaps, annot_colors, axes=None):
    """
    Creates a figure with four subplots showing the posterior probabilities of the comparisons.
    """
    posterior_matrix_smaller_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)
    posterior_matrix_equal_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)
    posterior_matrix_larger_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)
    posterior_matrix_decision_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)

    for i in range(result.posterior_matrix.shape[0]):
        for j in range(result.posterior_matrix.shape[1]):
            if isinstance(result.posterior_matrix.iloc[i, j], tuple):
                p_smaller, p_equal, p_larger = result.posterior_matrix.iloc[i, j]
                posterior_matrix_smaller_df.iloc[i, j] = p_smaller
                posterior_matrix_equal_df.iloc[i, j] = p_equal
                posterior_matrix_larger_df.iloc[i, j] = p_larger
                # this should be dropped, there is already a decision matrix
                decision = 1
                if result.decision_matrix.iloc[i, j]=='smaller':
                    decision = 2
                if result.decision_matrix.iloc[i, j]=='equal':
                    decision = 3
                if result.decision_matrix.iloc[i, j]=='larger':
                    decision = 4
                posterior_matrix_decision_df.iloc[i, j] = decision

    # drop first column and last row
    posterior_matrix_smaller_df = posterior_matrix_smaller_df.drop(columns=posterior_matrix_smaller_df.columns[0])
    posterior_matrix_smaller_df = posterior_matrix_smaller_df.drop(index=posterior_matrix_smaller_df.index[-1])
    posterior_matrix_equal_df = posterior_matrix_equal_df.drop(columns=posterior_matrix_equal_df.columns[0])
    posterior_matrix_equal_df = posterior_matrix_equal_df.drop(index=posterior_matrix_equal_df.index[-1])
    posterior_matrix_larger_df = posterior_matrix_larger_df.drop(columns=posterior_matrix_larger_df.columns[0])
    posterior_matrix_larger_df = posterior_matrix_larger_df.drop(index=posterior_matrix_larger_df.index[-1])
    posterior_matrix_decision_df = posterior_matrix_decision_df.drop(columns=posterior_matrix_decision_df.columns[0])
    posterior_matrix_decision_df = posterior_matrix_decision_df.drop(index=posterior_matrix_decision_df.index[-1])

    if axes is None:
        fig = plt.figure(figsize=(width, width))
        gs = mpl.gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        plt_created = True
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax4 = axes[3]

    if not isinstance(cmaps[3], mpl.colors.ListedColormap) or cmaps[3].N != 4:
        cmaps[3] = mpl.colormaps[cmaps[3]].resampled(4)
    decisions = ['', 'Inconclusive', '$A < B$', '$A = B$', '$A > B$', '']
    dec_fmt = mpl.ticker.FuncFormatter(lambda x, _: decisions[int(x)])

    _create_annotated_heatmap(ax1, posterior_matrix_smaller_df,
                              title_prefix="(a) ",
                              cmap=cmaps[0], cbarlabel="$P(A < B)$",
                              annot_color=annot_colors[0])
    _create_annotated_heatmap(ax2, posterior_matrix_equal_df,
                              title_prefix="(b) ",
                              cmap=cmaps[1], cbarlabel="$P(A = B)$",
                              annot_color=annot_colors[1])
    _create_annotated_heatmap(ax3, posterior_matrix_larger_df,
                              title_prefix="(c) ",
                              cmap=cmaps[2], cbarlabel="$P(A > B)$",
                              annot_color=annot_colors[2])
    _create_annotated_heatmap(ax4, posterior_matrix_decision_df,
                              title_prefix="(d) ",
                              cmap=cmaps[3], cbarlabel="Decision",
                              annot_color=None,
                              cbar_kw=dict(ticks=[0, 1, 2, 3, 4, 5], format=dec_fmt),
                              vmin=0.5, vmax=4.5)
    if plt_created:
        plt.tight_layout()


"""
Automated ranking of populations for ranking them. This is basically an implementation of Demsar's
Guidelines for the comparison of multiple classifiers. Details can be found in the description of the autorank function.
"""

import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from io import StringIO
from autorank._util import *

__all__ = ['autorank', 'plot_stats', 'plot_posterior_maps', 'create_report', 'latex_table', 'latex_report']

if 'text.usetex' in plt.rcParams and plt.rcParams['text.usetex']:
    raise UserWarning("plot_stats may fail if the matplotlib setting plt.rcParams['text.usetex']==True.\n"
                      "In case of failures you can try to set this value to False as follows:"
                      "plt.rc('text', usetex=False)")


def autorank(data, alpha=0.05, verbose=False, order='descending', approach='frequentist', rope=0.1, rope_mode='effsize',
             nsamples=50000, effect_size=None, force_mode=None, random_state=None, plot_order=None):
    """
    Automatically compares populations defined in a block-design data frame. Each column in the data frame contains
    the samples for one population. The data must not contain any NaNs. The data must have at least five measurements,
    i.e., rows. The current version is only reliable for less than 5000 measurements.

    The following approach is implemented by this function.

    - First all columns are checked with the Shapiro-Wilk test for normality. We use Bonferoni correction for these
      tests, i.e., alpha/len(data.columns).
    - If all columns are normal, we use Bartlett's test for homogeneity, otherwise we use Levene's test.
    - Based on the normality and the homogeneity, we select appropriate tests, effect sizes, and methods for determining
      the confidence intervals of the central tendency.

    If all columns are normal, we calculate:

    - The mean value as central tendency.
    - The empirical standard deviation as measure for the variance.
    - The confidence interval for the mean value.
    - The effect size in comparison to the highest mean value using Cohen's d.

    If at least one column is not normal, we calculate:

    - The median as central tendency.
    - The median absolute deviation from the median as measure for the variance.
    - The confidence interval for the median.
    - The effect size in comparison to the highest ranking approach using Cliff's delta.

    For the statistical tests, there are five variants:

    - If approach=='bayesian' we use a Bayesian signed rank test.
    - If there are two populations (columns) and both populations are normal, we use the paired t-test.
    - If there are two populations and at least one populations is not normal, we use Wilcoxon's signed rank test.
    - If there are more than two populations and all populations are normal and homoscedastic, we use repeated measures
      ANOVA with Tukey's HSD as post-hoc test.
    - If there are more than two populations and at least one populations is not normal or the populations are
      heteroscedastic, we use Friedman's test with the Nemenyi post-hoc test.

    # Parameters

    data (DataFrame):
        Each column contains a population and each row contains the paired measurements
        for the populations.

    alpha (float, default=0.05):
        Significance level. We internally use correction to ensure that all results (incl. confidence
        intervals) together fulfill this confidence level.

    verbose (bool, default=False):
        Prints decisions and p-values while running the autorank function to stdout.

    order (string, default='descending'):
        Determines the ordering central tendencies of the populations for the ranking. 'descending' results in higher
        ranks for larger values. 'ascending' results in higher ranks for smaller values.

    approach (string, default='frequentist'):
        With 'frequentist', a suitable frequentist statistical test is used (t-test, Wilcoxon signed rank test,
        ANOVA+Tukey's HSD, or Friedman+Nemenyi). With 'bayesian', the Bayesian signed ranked test is used.
        _(New in Version 1.1.0)_

    rope (float, default=0.01):
        Region of Practical Equivalence (ROPE) used for the bayesian analysis. The statistical analysis assumes that
        differences from the central tendency that are within the ROPE do not matter in practice. Therefore, such
        deviations may be considered to be equivalent. The ROPE is defined as an interval around the central tendency
        and the calculation of the interval is determined by the rope_mode parameter.
        _(New in Version 1.1.0)_

    rope_mode (string, default='effsize'):
        Method to calculate the size of the ROPE. With 'effsize', the ROPE is determined dynamically for each comparison
        of two populations as rope*effect_size, where effect size is either Cohen's d (normal data) or Akinshin's gamma
        (non-normal data). With 'absolute', the ROPE is defined using an absolute value that is used, i.e., the value of
        the rope parameter is used without any modification.
        _(New in Version 1.1.0)_

    nsamples (integer, default=50000):
        Number of samples used to estimate the posterior probabilities with the Bayesian signed rank test.
        _(New in Version 1.1.0)_

    effect_size (string, default=None):
        Effect size measure that is used for reporting. If None, the effect size is automatically selected as described
        in the flow chart. The following effect sizes are supported: "cohen_d", "cliff_delta", "akinshin_gamma".
        _(New in Version 1.1.0)_

    force_mode (string, default=None):
        Can be used to force autorank to use parametric or nonparametric frequentist tests. With 'parametric' you
        automatically get the t-test/repeated measures ANOVA. With 'nonparametric' you automatically get Wilcoxon's
        signed rank test/Friedman test. In case of Bayesian statistics, this parameter is used to override the automatic
        selection of the effect size measure, such that 'parametric' uses Cohen's d and 'nonparametric' uses Akinshin's,
        regardless of the normality of the data. If this parameter is None, the automatic selection is used.
        _(Support for Bayesian statistics added in Version 1.3.0)_

    random_state (integer, default=None):
        Seed for random state. Forwarded to Bayesian signed rank test to enable reproducible sampling and, thereby,
        reproducible results.
        _(New in Version 1.2.0)_

    plot_order (list):
        List with the order of the populations used for plotting, where reasonable (e.g., CI plots). If this is not none, this overrides the order parameter for visualizations.
        _(New in Version 1.3.0)_

    # Returns

    A named tuple of type RankResult with the following entries.

    rankdf (DataFrame):
        Ranked populations including statistics about the populations.

    pvalue (float):
        p-value of the omnibus test for the difference in central tendency between the populations. Not used with
        Bayesian statistics.

    omnibus (string):
       Omnibus test that is used for the test of a difference ein the central tendency.

    posthoc (string):
        Posthoc tests that was used. The posthoc test is performed even if the omnibus test is not significant. The
        results should only be used if the p-value of the omnibus test indicates significance. None in case of two
        populations and Bayesian statistics.

    cd (float):
        The critical distance of the Nemenyi posthoc test, if it was used. Otherwise None.

    all_normal (bool):
        True if all populations are normal, false if at least one is not normal.

    pvals_shapiro (list):
        p-values of the Shapiro-Wilk tests for normality sorted by the order of the input columns.

    homoscedastic (bool):
        True if populations are homoscedastic, false otherwise. None in case of Bayesian statistics.

    pval_homogeneity (float):
        p-value of the test for homogeneity. None in case of Bayesian statistics.

    homogeneity_test (string):
        Test used for homogeneity. Either 'bartlet' or 'levene'.

    alpha (float):
        Family-wise significant level. Same as input parameter.

    alpha_normality (float):
        Corrected alpha that is used for tests for normality.

    num_samples (int):
        Number of samples within each population.

    order (string):
        Order of the central tendencies used for ranking.

    sample_matrix (DataFrame):
        Matrix with SignedRankTest objects from package baycomp. Can be used to do further analysis, e.g. to generate
        plots using the built-in plot() method of baycomp. For a detailed description of methods and parameters, see
        the documentation of baycomp: https://baycomp.readthedocs.io/en/latest/classes.html#multiple-data-sets
        _(New in Version 1.2.0)_

    posterior_matrix (DataFrame):
        Matrix with the pair-wise posterior probabilities estimated with the Bayesian signed ranked test. The matrix
        is a square matrix with the populations sorted by their central tendencies as rows and columns. The value of
        the matrix in the i-th row and the j-th column contains a 3-tuple (p_smaller, p_equal, p_greater) such that
        p_smaller is the probability that the population in column j is smaller than the population in row i, p_equal
        that both populations are equal, and p_larger that population j is larger than population i. If rope==0.0, the
        matrix contains only 2-tuples (p_smaller, p_greater) because equality is not possible without a ROPE.
        _(New in Version 1.1.0)_

    decision_matrix (DataFrame):
        Matrix with the pair-wise decisions made with the Bayesian signed ranked test. The matrix is a square matrix
        with the populations sorted by their central tendencies as rows and columns. The value of
        the matrix in the i-th row and the j-th column contains the value 'smaller' if the population in column j is
        significantly larger than the population in row i, 'equal' is both populations are equivalent (i.e., have no
        practically relevant difference), 'larger' if the population in column j is larger than the population in
        column i, and 'inconclusive' if the statistical analysis is did not yield a definitive result.
        _(New in Version 1.1.0)_

    rope (float):
        Region of Practical Equivalence (ROPE). Same as input parameter.
        _(New in Version 1.1.0)_

    rope_mode (string):
        Mode for calculating the ROPE. Same as input parameter.
        _(New in Version 1.1.0)_

    effect_size (string):
        Effect size measure that is used for reporting. Same as input parameter.

    force_mode (string):
        If not None, this is the force mode that was used to select the tests. Either 'parametric' or 'nonparametric'.

    plot_order (list):
        If not None, this is the fixed order that is used for plotting, where possible. Otherwise None.
        _(New in Version 1.3.0)_
    """

    # validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if len(data.columns) < 2:
        raise ValueError('requires at least two classifiers (i.e., columns)')
    if len(data) < 5:
        raise ValueError('requires at least five performance estimations (i.e., rows)')

    if not isinstance(alpha, float):
        raise TypeError('alpha must be a float')
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError('alpha must be in the open interval (0.0,1.0)')

    if not isinstance(verbose, bool):
        raise TypeError('verbose must be bool')

    if not isinstance(order, str):
        raise TypeError('order must be str')
    if order not in ['ascending', 'descending']:
        raise ValueError("order must be either 'ascending' or 'descending'")

    if not isinstance(approach, str):
        raise TypeError('approach must be str')
    if approach not in ['frequentist', 'bayesian']:
        raise ValueError("approach must be either 'frequentist' or 'bayesian'")

    if not isinstance(rope, (int, float)):
        raise TypeError('rope must be a numeric')
    if rope < 0.0:
        raise ValueError('rope must be positive')

    if not isinstance(rope_mode, str):
        raise TypeError('rope_mode must be str')
    if rope_mode not in ['effsize', 'absolute']:
        raise ValueError("rope_mode must be either 'effsize' or 'absolute'")

    if not isinstance(nsamples, int):
        raise TypeError('nsamples must be an integer')
    if nsamples < 1:
        raise ValueError('nsamples must be positive')

    if effect_size is not None:
        if not isinstance(effect_size, str):
            raise TypeError("effect_size must be a string")
        if effect_size not in ['cohen_d', 'cliff_delta', 'akinshin_gamma']:
            raise ValueError("effect_size must be None or one of the following: 'cohen_d', 'cliff_delta', "
                             "'akinshin_gamma'")

    if force_mode is not None:
        if not isinstance(force_mode, str):
            raise TypeError("force mode must be a string")
        if force_mode not in ['parametric', 'nonparametric']:
            raise ValueError("force_mode must be None or one of the following 'parametric', 'nonparametric'")

    if force_mode is not None and approach=='frequentist':
        print("Tests for normality and homoscedacity are ignored for test selection, forcing %s tests" % force_mode)

    if plot_order is not None:
        if not isinstance(plot_order, list):
            raise TypeError("plot_order must be a list")
        if len(plot_order) != len(data.columns):
            raise ValueError("plot_order must have the same length as the number of columns in data")
        if not all(isinstance(x, str) for x in plot_order):
            raise TypeError("plot_order must contain only strings")
        if not all(x in data.columns for x in plot_order):
            raise ValueError("plot_order must contain only columns from data")
        if len(set(plot_order)) != len(plot_order):
            raise ValueError("plot_order must not contain duplicates (not supported for data frames with duplicate column names)")

    # ensure that the index is not named or a MultiIndex
    # this trips up some internal functions (e.g., Anova (see issue #16))
    if data.index.name is not None or isinstance(data.index, pd.MultiIndex):
        data = data.reset_index(drop=True)

    # ensure that index and columns are not named
    # this also trips up some internal functions (e.g., Anova (see issue #37))
    if data.index.name is not None:
        data = data.rename_axis(None, axis=0)
    if data.columns.name is not None:
        data = data.rename_axis(None, axis=1)

    # Bonferoni correction for normality tests
    alpha_normality = alpha / len(data.columns)
    all_normal, pvals_shapiro = test_normality(data, alpha_normality, verbose)

    # Select appropriate tests
    if approach == 'frequentist':
        # homogeneity needs only to be checked for frequentist approach
        if all_normal:
            if verbose:
                print("Using Bartlett's test for homoscedacity of normally distributed data")
            homogeneity_test = 'bartlett'
            pval_homogeneity = stats.bartlett(*data.transpose().values).pvalue
        else:
            if verbose:
                print("Using Levene's test for homoscedacity of non-normal data.")
            homogeneity_test = 'levene'
            pval_homogeneity = stats.levene(*data.transpose().values).pvalue
        var_equal = pval_homogeneity >= alpha
        if verbose:
            if var_equal:
                print("Fail to reject null hypothesis that all variances are equal "
                      "(p=%f>=%f)" % (pval_homogeneity, alpha))
            else:
                print("Rejecting null hypothesis that all variances are equal (p=%f<%f)" % (pval_homogeneity, alpha))

        if len(data.columns) == 2:
            res = rank_two(data, alpha, verbose, all_normal, order, effect_size, force_mode)
        else:
            if (force_mode is not None and force_mode=='parametric') or \
               (force_mode is None and all_normal and var_equal):
                res = rank_multiple_normal_homoscedastic(data, alpha, verbose, order, effect_size, force_mode)
            else:
                res = rank_multiple_nonparametric(data, alpha, verbose, all_normal, order, effect_size, force_mode)
        # need to reorder pvals here (see issue #7)
        pvals_shapiro = [pvals_shapiro[pos] for pos in res.reorder_pos]
        return RankResult(res.rankdf, res.pvalue, res.cd, res.omnibus, res.posthoc, all_normal, pvals_shapiro,
                          var_equal, pval_homogeneity, homogeneity_test, alpha, alpha_normality, len(data), None, None,
                          None, None, None, res.effect_size, force_mode, plot_order)
    elif approach == 'bayesian':
        res = rank_bayesian(data, alpha, verbose, all_normal, order, rope, rope_mode, nsamples, effect_size, random_state, force_mode)
        # need to reorder pvals here (see issue #7)
        pvals_shapiro = [pvals_shapiro[pos] for pos in res.reorder_pos]
        return RankResult(res.rankdf, None, None, 'bayes', 'bayes', all_normal, pvals_shapiro, None, None, None, alpha,
                          alpha_normality, len(data), res.sample_matrix, res.posterior_matrix, res.decision_matrix, rope,
                          rope_mode, res.effect_size, force_mode, plot_order)


def plot_stats(result, *, allow_insignificant=False, ax=None, width=None):
    """
    Creates a plot that supports the analysis of the results of the statistical test. The plot depends on the
    statistical test that was used.

    - Creates a Confidence Interval (CI) plot for a paired t-test between two normal populations. The confidence
     intervals are calculated with Bonferoni correction, i.e., a confidence level of alpha/2.
    - Creates a CI plot for Tukey's HSD as post-hoc test with the confidence intervals calculated using the HSD approach
     such that the family wise significance is alpha.
    - Creates Critical Distance (CD) diagrams for the Nemenyi post-hoc test. CD diagrams visualize the mean ranks of
     populations. Populations that are not significantly different are connected by a horizontal bar.

    This function raises a ValueError if the omnibus test did not detect a significant difference. The allow_significant
    parameter allows the suppression of this exception and forces the creation of the plots.

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    allow_insignificant (bool, default=False):
        Forces plotting even if results are not significant.

    ax (Axis, default=None):
        Matplotlib axis to which the results are added. A new figure with a single axis is created if None.

    width (float, default=None):
        Specifies the width of the created plot is not None. By default, we use a width of 6. The height is
        automatically determined, based on the type of plot and the number of populations. This parameter is ignored if
        ax is not None.

    # Return

    Axis with the plot. None if no plot was generated.
    """
    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")

    if result.omnibus == 'bayes':
        raise ValueError("plotting results of bayesian analysis not yet supported.")

    if result.pvalue >= result.alpha and not allow_insignificant:
        raise ValueError(
            "result is not significant and results of the plot may be misleading. If you want to create the plot "
            "regardless, use the allow_insignificant parameter to suppress this exception.")

    if ax is not None and width is not None:
        warnings.warn('width may be ignored because ax is defined.')
    if width is None:
        width = 6

    if result.omnibus == 'ttest':
        ax = ci_plot(result, True, ax, width)
    elif result.omnibus == 'wilcoxon':
        warnings.warn('No plot to visualize statistics for Wilcoxon test available. Doing nothing.')
    elif result.posthoc == 'tukeyhsd':
        ax = ci_plot(result, True, ax, width)
    elif result.posthoc == 'nemenyi':
        ax = cd_diagram(result, False, ax, width)
    return ax


def plot_posterior_maps(result, *, width=None, cmaps=None, annot_colors=None, axes=None, ):
    """
    Creates a posterior map plot for the results of the Bayesian signed rank test. The posterior map shows the
    posterior probabilities of the pair-wise comparisons between the populations.
    _(New in Version 1.3.0)_

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    axes (list, default=None):
        List of matplotlib axes to which the results are added. A new figure with a single axis is created if None.
        If there are more than one axes, they are used to create multiple subplots.

    width (float, default=None):
        Specifies the width of the created plot is not None. By default, we use a width of 10. The height is
        automatically set to the same value of width, since the maps should be square. This parameter is ignored if
        axes is not None.

    cmaps (list, default=['Blues', 'Oranges', 'Greys', custom_cmap]):
        Colormaps used for the posterior maps. The default custom_cmap is used for the decisions with four colors matching the
        colors of the posterior maps (+ one color for inconclusive).

    annot_colors (list, default=[("black", "white"), ("black", "white"), ("black", "white"), ("black", "white")]):
        Colors used for the annotations in the posterior maps. The first color is used for less intensive backgrounds,
        the second color for intensive backgrounds.
    """

    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")
    if width is None:
        width = 10
    if not isinstance(width, int) and not isinstance(width, float):
        raise TypeError("width must be a number or None")
    if width <= 0:
        raise ValueError("width must be positive")
    if cmaps is None:
        # Define the colors
        colors = ['whitesmoke', 'mediumseagreen', 'lightgrey', 'orangered']
        # Create the colormap
        cmap = mpl.colors.ListedColormap(colors, name='custom_colormap')
        cmaps = ['Greens', 'Oranges', 'Greys', cmap]
    if not isinstance(cmaps, list):
        raise TypeError("cmaps must be a list of colormaps or None")
    if len(cmaps) != 4:
        raise ValueError("cmaps must have exactly 4 elements")
    if annot_colors is None:
        annot_colors = [("black", "white"), ("black", "white"), ("black", "white"), ("black", "white")]
    if not isinstance(annot_colors, list):
        raise TypeError("annot_colors must be a list of colors or None")
    if len(annot_colors) != 4:
        raise ValueError("annot_colors must have exactly 4 elements")
    if axes is not None:
        if not isinstance(axes, list):
            raise TypeError("axes must be a list of matplotlib axes or None")
        if len(axes) != 4:
            raise ValueError("axes must have exactly 4 elements")

    if result.omnibus != 'bayes':
        raise ValueError("plot_posterior_maps can only be used with Bayesian analysis results.")

    posterior_maps(result, axes=axes, width=width, cmaps=cmaps, annot_colors=annot_colors)


def create_report(result, *, decimal_places=3):
    """
    Prints a report about the statistical analysis.

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    decimal_places (int, default=3):
        Number of decimal places that are used for the report.
    """

    # TODO add effect sizes to multiple comparisons.
    def single_population_string(population, with_stats=False, pop_pval=None, with_rank=True):
        if pop_pval is not None:
            return "%s (p=%.*f)" % (population, decimal_places, pop_pval)
        if with_stats:
            halfwidth = (result.rankdf.at[population, 'ci_upper'] - result.rankdf.at[population, 'ci_lower']) / 2
            mystats = []
            if (result.force_mode is not None and result.force_mode=='parametric') or \
                    (result.force_mode is None and result.all_normal):
                mystats.append("M=%.*f+-%.*f" % (decimal_places, result.rankdf.at[population, 'mean'],
                                                 decimal_places, halfwidth))
                mystats.append("SD=%.*f" % (decimal_places, result.rankdf.at[population, 'std']))
            else:
                mystats.append("MD=%.*f+-%.*f" % (decimal_places, result.rankdf.at[population, 'median'],
                                                  decimal_places, halfwidth))
                mystats.append("MAD=%.*f" % (decimal_places, result.rankdf.at[population, 'mad']))
            if with_rank:
                mystats.append("MR=%.*f" % (decimal_places, result.rankdf.at[population, 'meanrank']))
            return "%s (%s)" % (population, ", ".join(mystats))
        else:
            return str(population)

    def create_population_string(populations, with_stats=False, pop_pvals=None, with_rank=False):
        if isinstance(populations, str):
            populations = [populations]
        population_strings = []
        for index, population in enumerate(populations):
            if pop_pvals is not None:
                cur_pval = pop_pvals[index]
            else:
                cur_pval = None
            population_strings.append(single_population_string(population, with_stats, cur_pval, with_rank))
        if len(populations) == 1:
            popstr = population_strings[0]
        elif len(populations) == 2:
            popstr = " and ".join(population_strings)
        else:
            popstr = ", ".join(population_strings[:-1]) + ", and " + population_strings[-1]
        return popstr

    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")

    print("The statistical analysis was conducted for %i populations with %i paired samples." % (len(result.rankdf),
                                                                                                 result.num_samples))
    print("The family-wise significance level of the tests is alpha=%.*f." % (decimal_places, result.alpha))

    if result.all_normal:
        not_normal = []
        min_pvalue = min(result.pvals_shapiro)
        print("We failed to reject the null hypothesis that the population is normal for all populations "
              "(minimal observed p-value=%.*f). Therefore, we assume that all populations are "
              "normal." % (decimal_places, min_pvalue))
    else:
        not_normal = []
        pvals = []
        normal = []
        for i, pval in enumerate(result.pvals_shapiro):
            if pval < result.alpha_normality:
                not_normal.append(result.rankdf.index[i])
                pvals.append(pval)
            else:
                normal.append(result.rankdf.index[i])
        if len(not_normal) == 1:
            population_term = 'population'
        else:
            population_term = 'populations'
        print("We rejected the null hypothesis that the population is normal for the %s %s. "
              "Therefore, we assume that not all populations are "
              "normal." % (population_term, create_population_string(not_normal, pop_pvals=pvals)))

    if result.omnibus == 'bayes':
        if (result.force_mode is not None and result.force_mode=='parametric') or (result.force_mode is None and result.all_normal):
            central_tendency = 'mean value'
            central_tendency_long = 'mean value (M)'
            variability = 'standard deviation (SD)'
            effect_size = 'd'
        else:
            central_tendency = 'median'
            central_tendency_long = 'median (MD)'
            variability = 'median absolute deviation (MAD)'
            effect_size = 'gamma'
        print(
            "We used a bayesian signed rank test to determine differences between the mean values of the "
            "populations and report the %s and the %s for each population. We distinguish "
            "between populations being pair-wise smaller, equal, or larger and make a decision for one "
            "of these cases if we estimate that the posterior probability is at least "
            "alpha=%.*f." % (central_tendency_long, variability, decimal_places, result.alpha))
        if result.rope_mode == 'effsize':
            print(
                'We used the effect size to define the region of practical equivalence (ROPE) around the %s '
                'dynamically as %.*f*%s.' % (central_tendency, decimal_places, result.rope, effect_size))
        else:
            print(
                'We used a fixed value of %.*f to define the region of practical equivalence (ROPE) around the '
                '%s.' % (decimal_places, result.rope, central_tendency))
        decision_set = set(result.rankdf['decision'])
        decision_set.remove('NA')
        if {'inconclusive'} == decision_set:
            print("We failed to find any conclusive evidence for differences between the populations "
                  "%s." % create_population_string(result.rankdf.index, with_stats=True))
        elif {'equal'} == decision_set:
            print(
                "All populations are equal, i.e., the are no significant and practically relevant differences "
                "between the populations %s." % create_population_string(result.rankdf.index,
                                                                         with_stats=True))
        elif {'equal', 'inconclusive'} == decision_set:
            print(
                "The populations %s are all either equal or the results of the analysis are inconclusive." % create_population_string(result.rankdf.index, with_stats=True))
            print(result.decision_matrix)
        else:
            print("We found significant and practically relevant differences between the populations "
                  "%s." % create_population_string(result.rankdf.index, with_stats=True))
            for i in range(len(result.rankdf)):
                if len(result.rankdf.index[result.decision_matrix.iloc[i, :] == 'smaller']) > 0:
                    print('The %s of the population %s is larger than of the populations '
                          '%s.' % (central_tendency, result.rankdf.index[i],
                                   create_population_string(
                                       result.rankdf.index[
                                           result.decision_matrix.iloc[i, :] == 'smaller'])))
            equal_pairs = []
            for i in range(len(result.rankdf)):
                for j in range(i + 1, len(result.rankdf)):
                    if result.decision_matrix.iloc[i, j] == 'equal':
                        equal_pairs.append(result.rankdf.index[i] + ' and ' + result.rankdf.index[j])
            if len(equal_pairs) > 0:
                equal_pairs_str = create_population_string(equal_pairs).replace(',', ';')
                print('The following pairs of populations are equal: %s.' % equal_pairs_str)
            if 'inconclusive' in set(result.rankdf['decision']):
                print('All other differences are inconclusive.')
    elif len(result.rankdf) == 2:
        print("No check for homogeneity was required because we only have two populations.")
        if result.effect_size == 'cohen_d':
            effect_size = 'd'
        elif result.effect_size == 'cliff_delta':
            effect_size = 'delta'
        elif result.effect_size == 'akinshin_gamma':
            effect_size = 'gamma'
        else:
            raise ValueError('unknown effect size method, this should not be possible: %s' % result.effect_size)
        if result.omnibus == 'ttest':
            larger = np.argmax(result.rankdf['mean'].values)
            smaller = int(bool(larger - 1))
            if result.all_normal:
                print("Because we have only two populations and both populations are normal, we use the t-test to "
                      "determine differences between the mean values of the populations and report the mean value (M)"
                      "and the standard deviation (SD) for each population. ")
            else:
                if len(not_normal) == 1:
                    notnormal_str = 'one of them is'
                else:
                    notnormal_str = 'both of them are'
                print("Because we have only two populations and %s not normal, we use should Wilcoxon's signed rank "
                      "test to determine the differences in the central tendency and report the median (MD) and the "
                      "median absolute deviation (MAD) for each population. However, the user decided to force the "
                      "use of the t-test which assumes normality of all populations and we report the mean value (M) "
                      "and the standard deviation (SD) for each population." % notnormal_str)
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of the paired t-test that the mean values of "
                      "the populations %s are are equal. Therefore, we "
                      "assume that there is no statistically significant difference between the mean values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of the paired t-test that the mean values of the "
                      "populations %s are "
                      "equal. Therefore, we assume that the mean value of %s is "
                      "significantly larger than the mean value of %s with a %s effect size (%s=%.*f)."
                      % (decimal_places, result.pvalue,
                         create_population_string(result.rankdf.index, with_stats=True),
                         result.rankdf.index[larger], result.rankdf.index[smaller],
                         result.rankdf.magnitude.iloc[larger], effect_size, decimal_places, result.rankdf.effect_size.iloc[larger]))
        elif result.omnibus == 'wilcoxon':
            larger = np.argmax(result.rankdf['median'].values)
            smaller = int(bool(larger - 1))
            if result.all_normal:
                print("Because we have only two populations and both populations are normal, we should use the t-test "
                      "to determine differences between the mean values of the populations and report the mean value "
                      "(M) and the standard deviation (SD) for each population. However, the user decided to force the "
                      "use of the less powerful Wilcoxon signed rank test and we report the median (MD) and the median "
                      "absolute devivation (MAD) for each population.")
            else:
                if len(not_normal) == 1:
                    notnormal_str = 'one of them is'
                else:
                    notnormal_str = 'both of them are'
                print("Because we have only two populations and %s not normal, we use Wilcoxon's signed rank test to "
                      "determine the differences in the central tendency and report the median (MD) and the median "
                      "absolute deviation (MAD) for each population." % notnormal_str)
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of Wilcoxon's signed rank test that "
                      "population %s is not greater than population %s . Therefore, we "
                      "assume that there is no statistically significant difference between the medians of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index[larger], with_stats=True),
                                        create_population_string(result.rankdf.index[smaller], with_stats=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of Wilcoxon's signed rank test that population "
                      "%s is not greater than population %s. Therefore, we assume "
                      "that the median of %s is "
                      "significantly larger than the median value of %s with a %s effect size (%s=%.*f)."
                      % (decimal_places, result.pvalue,
                         create_population_string(result.rankdf.index[larger], with_stats=True),
                         create_population_string(result.rankdf.index[smaller], with_stats=True),
                         result.rankdf.index[larger], result.rankdf.index[smaller],
                         result.rankdf.magnitude.iloc[larger], effect_size, decimal_places, result.rankdf.effect_size.iloc[larger]))
        else:
            raise ValueError('Unknown omnibus test for difference in the central tendency: %s' % result.omnibus)
    else:
        if result.all_normal:
            if result.homoscedastic:
                print("We applied Bartlett's test for homogeneity and failed to reject the null hypothesis "
                      "(p=%.*f) that the data is homoscedastic. Thus, we assume that our data is "
                      "homoscedastic." % (decimal_places, result.pval_homogeneity))
            else:
                print("We applied Bartlett's test for homogeneity and reject the null hypothesis (p=%.*f) that the "
                      "data is homoscedastic. Thus, we assume that our data is "
                      "heteroscedastic." % (decimal_places, result.pval_homogeneity))

        if result.omnibus == 'anova':
            if result.all_normal and result.homoscedastic:
                print("Because we have more than two populations and all populations are normal and homoscedastic, we "
                      "use repeated measures ANOVA as omnibus "
                      "test to determine if there are any significant differences between the mean values of the "
                      "populations. If the results of the ANOVA test are significant, we use the post-hoc Tukey HSD "
                      "test to infer which differences are significant. We report the mean value (M) and the standard "
                      "deviation (SD) for each population. Populations are significantly different if their confidence "
                      "intervals are not overlapping.")
            else:
                if result.all_normal:
                    print(
                        "Because we have more than two populations and the populations are normal but heteroscedastic, "
                        "we should use the non-parametric Friedman test "
                        "as omnibus test to determine if there are any significant differences between the mean values "
                        "of the populations. However, the user decided to force the use of "
                        "repeated measures ANOVA as omnibus test which assume homoscedascity to determine if there are "
                        "any significant difference between the mean values of the populations. If the results of the "
                        "ANOVA test are significant, we use the post-hoc Tukey HSD test to infer which differences are "
                        "significant. We report the mean value (M) and the standard deviation (SD) for each "
                        "population. Populations are significantly different if their confidence intervals are not "
                        "overlapping.")
                else:
                    if len(not_normal) == 1:
                        notnormal_str = 'one of them is'
                    else:
                        notnormal_str = 'some of them are'
                    print("Because we have more than two populations and the populations and %s not normal, "
                          "we should use the non-parametric Friedman test "
                          "as omnibus test to determine if there are any significant differences between the median "
                          "values of the populations and report the median (MD) and the median absolute deviation "
                          "(MAD). However, the user decided to force the use of repeated measures ANOVA as omnibus "
                          "test which assume homoscedascity to determine if there are any significant difference "
                          "between the mean values of the populations. If the results of the ANOVA test are "
                          "significant, we use the post-hoc Tukey HSD test to infer which differences are "
                          "significant. We report the mean value (M) and the standard deviation (SD) for each "
                          "population. Populations are significantly different if their confidence intervals are not "
                          "overlapping." % (notnormal_str))
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of the repeated measures ANOVA that there is "
                      "no difference between the mean values of the populations %s. Therefore, we "
                      "assume that there is no statistically significant difference between the mean values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of the repeated measures ANOVA that there is "
                      "no difference between the mean values of the populations %s. Therefore, we "
                      "assume that there is a statistically significant difference between the mean values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True)))
                meanranks, names, groups = get_sorted_rank_groups(result, False)
                if len(groups) == 0:
                    print("Based on post-hoc Tukey HSD test, we assume that all differences between the populations "
                          "are significant.")
                else:
                    groupstrs = []
                    for group_range in groups:
                        group = range(group_range[0], group_range[1] + 1)
                        if len(group) == 1:
                            cur_groupstr = names[group[0]]
                        elif len(group) == 2:
                            cur_groupstr = " and ".join([names[pop] for pop in group])
                        else:
                            cur_groupstr = ", ".join([names[pop] for pop in group[:-1]]) + ", and " + names[group[-1]]
                        groupstrs.append(cur_groupstr)
                    print("Based post-hoc Tukey HSD test, we assume that there are no significant differences within "
                          "the following groups: %s. All other differences are significant." % ("; ".join(groupstrs)))
                print()
        elif result.omnibus == 'friedman':
            if result.all_normal and result.homoscedastic:
                print("Because we have more than two populations and all populations are normal and homoscedastic, we "
                      "should use repeated measures ANOVA as omnibus "
                      "test to determine if there are any significant differences between the mean values of the "
                      "populations. However, the user decided to force the use of the less powerful Friedman test as "
                      "omnibus test to determine if there are any significant differences between the mean values "
                      "of the populations. We report the mean value (M), the standard deviation (SD) and the mean rank "
                      "(MR) among all populations over the samples. Differences between populations are significant, "
                      "if the difference of the mean rank is greater than the critical distance CD=%.*f of the Nemenyi "
                      "test." % (decimal_places, result.cd))
            elif result.all_normal:
                print("Because we have more than two populations and the populations are normal but heteroscedastic, "
                      "we use the non-parametric Friedman test "
                      "as omnibus test to determine if there are any significant differences between the mean values "
                      "of the populations. We use the post-hoc Nemenyi test to infer which differences are "
                      "significant. We report the mean value (M), the standard deviation (SD) and the mean rank (MR) "
                      "among all populations over the samples. Differences between populations are significant, if the "
                      "difference of the mean rank is greater than the critical distance CD=%.*f of the Nemenyi "
                      "test." % (decimal_places, result.cd))
            else:
                if len(not_normal) == 1:
                    notnormal_str = 'one of them is'
                else:
                    notnormal_str = 'some of them are'
                print("Because we have more than two populations and the populations and %s not normal, "
                      "we use the non-parametric Friedman test "
                      "as omnibus test to determine if there are any significant differences between the median values "
                      "of the populations. We use the post-hoc Nemenyi test to infer which differences are "
                      "significant. We report the median (MD), the median absolute deviation (MAD) and the mean rank "
                      "(MR) among all populations over the samples. Differences between populations are significant, "
                      "if the difference of the mean rank is greater than the critical distance CD=%.*f of the Nemenyi "
                      "test." % (notnormal_str, decimal_places, result.cd))
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of the Friedman test that there is no "
                      "difference in the central tendency of the populations %s. Therefore, we "
                      "assume that there is no statistically significant difference between the median values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True, with_rank=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of the Friedman test that there is no "
                      "difference in the central tendency of the populations %s. Therefore, we "
                      "assume that there is a statistically significant difference between the median values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True, with_rank=True)))
                meanranks, names, groups = get_sorted_rank_groups(result, False)
                if len(groups) == 0:
                    print("Based on the post-hoc Nemenyi test, we assume that all differences between the populations "
                          "are significant.")
                else:
                    groupstrs = []
                    for group_range in groups:
                        group = range(group_range[0], group_range[1] + 1)
                        if len(group) == 1:
                            cur_groupstr = names[group[0]]
                        elif len(group) == 2:
                            cur_groupstr = " and ".join([names[pop] for pop in group])
                        else:
                            cur_groupstr = ", ".join([names[pop] for pop in group[:-1]]) + ", and " + names[group[-1]]
                        groupstrs.append(cur_groupstr)
                    print("Based on the post-hoc Nemenyi test, we assume that there are no significant differences "
                          "within the following groups: %s. All other differences are "
                          "significant." % ("; ".join(groupstrs)))
        else:
            raise ValueError('Unknown omnibus test for difference in the central tendency: %s' % result.omnibus)


def latex_table(result, *, decimal_places=3, label=None, effect_size_relation="best", posterior_relation="best"):
    """
    Creates a latex table from the results dataframe of the statistical analysis.

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    decimal_places (int, default=3):
        Number of decimal places that are used for the report.

    label (str, default=None):
        Label of the table. Defaults to 'tbl:stat_results' if None.

    effect_size_relation (str, default="best"):
        Specifies which effect size relation is used in the table. Can be "best", "above", or both.
        If "best", the effect size is compute in relation to the best-ranked value.
        If "above", the effect size is computed in relation to the value above in the row above.
        With "both", both the best and the above are included in the table.
        _(New in Version 1.3.0)_

    posterior_relation (str, default="best"):
        Specifies which posterior relation is used in the table. Can be "best", "above", or both.
        If "best", the posterior is computed in relation to the best-ranked value.
        If "above", the posterior is computed in relation to the value above in the row above.
        With "both", both the best and the above are included in the table.
        _(New in Version 1.3.0)_

    """
    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")
    if effect_size_relation not in {'best', 'above', 'both'}:
        raise ValueError("effect_size_relation must be one of 'best', 'above', or 'both'.")
    if posterior_relation not in {'best', 'above', 'both'}:
        raise ValueError("posterior_relation must be one of 'best', 'above', or 'both'.")

    if label is None:
        label = 'tbl:stat_results'

    table_df = result.rankdf.copy(deep=True)
    columns = table_df.columns.to_list()
    if result.omnibus != 'bayes' and result.pvalue >= result.alpha or \
       result.omnibus == 'bayes' and len({'smaller', 'larger'}.intersection(set(result.rankdf['decision']))) == 0:
        columns.remove('effect_size')
        columns.remove('magnitude')
    if result.posthoc == 'tukeyhsd':
        columns.remove('meanrank')
    if result.omnibus == 'bayes':
        table_df.at[table_df.index[0], 'decision'] = '-'
        table_df.at[table_df.index[0], 'decision_above'] = '-'
    columns.insert(columns.index('ci_lower'), 'CI')
    columns.remove('ci_lower')
    columns.remove('ci_upper')
    rename_map = {}
    if result.effect_size == 'cohen_d':
        if effect_size_relation == 'best':
            rename_map['effect_size'] = '$d$'
            columns.remove('effect_size_above')
        elif effect_size_relation == 'above':
            rename_map['effect_size_above'] = '$d$'
            columns.remove('effect_size')
        elif effect_size_relation == 'both':
            rename_map['effect_size'] = '$d$ (best)'
            rename_map['effect_size_above'] = '$d$ (above)'
    elif result.effect_size == 'cliff_delta':
        if effect_size_relation == 'best':
            rename_map['effect_size'] = r'D-E-L-T-A'
            columns.remove('effect_size_above')
        elif effect_size_relation == 'above':
            rename_map['effect_size_above'] = r'D-E-L-T-A'
            columns.remove('effect_size')
        elif effect_size_relation == 'both':
            rename_map['effect_size'] = r'D-E-L-T-A (best)'
            rename_map['effect_size_above'] = r'D-E-L-T-A (above)'
    elif result.effect_size == 'akinshin_gamma':
        if effect_size_relation == 'best':
            rename_map['effect_size'] = r'G-A-M-M-A'
            columns.remove('effect_size_above')
        elif effect_size_relation == 'above':
            rename_map['effect_size_above'] = r'G-A-M-M-A'
            columns.remove('effect_size')
        elif effect_size_relation == 'both':
            rename_map['effect_size'] = r'G-A-M-M-A (best)'
            rename_map['effect_size_above'] = r'G-A-M-M-A (above)'
    if effect_size_relation == 'best':
        rename_map['magnitude'] = 'Magnitude'
        columns.remove('magnitude_above')
    elif effect_size_relation == 'above':
        rename_map['magnitude_above'] = 'Magnitude'
        columns.remove('magnitude')
    elif effect_size_relation == 'both':
        rename_map['magnitude'] = 'Magnitude (best)'
        rename_map['magnitude_above'] = 'Magnitude (above)'
    rename_map['mad'] = 'MAD'
    rename_map['median'] = 'MED'
    rename_map['meanrank'] = 'MR'
    rename_map['mean'] = 'M'
    rename_map['std'] = 'SD'
    if posterior_relation == 'best':
        rename_map['decision'] = 'Decision'
        if 'decision_above' in columns:
            columns.remove('decision_above')
            columns.remove('p_equal_above')
            columns.remove('p_smaller_above')
    elif posterior_relation == 'above':
        rename_map['decision_above'] = 'Decision'
        if 'decision' in columns:
            columns.remove('decision')
            columns.remove('p_equal')
            columns.remove('p_smaller')
    elif posterior_relation == 'both':
        rename_map['decision'] = 'Decision (best)'
        if 'decision_above' in columns:
            rename_map['decision_above'] = 'Decision (above)'
    format_string = '[{0[ci_lower]:.' + str(decimal_places) + 'f}, {0[ci_upper]:.' + str(decimal_places) + 'f}]'
    table_df['CI'] = table_df.agg(format_string.format, axis=1)
    table_df = table_df[columns]
    table_df = table_df.rename(rename_map, axis='columns')

    float_format = lambda x: ("{:0." + str(decimal_places) + "f}").format(x) if not np.isnan(x) else '-'
    table_string = table_df.to_latex(float_format=float_format, na_rep='-').strip()
    table_string = table_string.replace('D-E-L-T-A', r'$\delta$')
    table_string = table_string.replace('G-A-M-M-A', r'$\gamma$')
    if posterior_relation == 'best':
        table_string = table_string.replace(r'p_equal', r'$P(\textit{equal})$')
        table_string = table_string.replace(r'p_smaller', r'$P(\textit{smaller})$')
    elif posterior_relation == 'above':
        table_string = table_string.replace(r'p_equal_above', r'$P(\textit{equal})$')
        table_string = table_string.replace(r'p_smaller_above', r'$P(\textit{smaller})$')
    elif posterior_relation == 'both':
        table_string = table_string.replace(r'p_equal_above', r'$P(\textit{equal})$ (above)')
        table_string = table_string.replace(r'p_smaller_above', r'$P(\textit{smaller})$ (above)')
        table_string = table_string.replace(r'p_equal', r'$P(\textit{equal})$ (best)')
        table_string = table_string.replace(r'p_smaller', r'$P(\textit{smaller})$ (best)')
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(table_string)
    print(r"\caption{Summary of populations}")
    print(r"\label{%s}" % label)
    print(r"\end{table}")


def latex_report(result, *, decimal_places=3, prefix="", generate_plots=True, figure_path="", complete_document=True):
    """
    Creates a latex report of the statistical analysis.

    # Parameters

    result (AutoRank):
        Should be the return value the autorank function.

    decimal_places (int, default=3):
        Number of decimal places that are used for the report.

    prefix (str, default=""):
        Prefix that is added before all labels and plot file names.

    generate_plots (bool, default=True):
        Decides if plots are generated, if the results are statistically significant.

    figure_path (str, default=""):
        Path where the plots shall be written to. Ignored if generate_plots is False.

    complete_document (bool, default=True):
        Generates a complete latex document if true. Otherwise only a single section is generated.
    """
    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")

    if complete_document:
        print(r"\documentclass{article}")
        print()
        print(r"\usepackage{graphicx}")
        print(r"\usepackage{booktabs}")
        print()
        print(r"\begin{document}")
        print()

    print(r"\section{Results}")
    print(r"\label{sec:%sresults}" % prefix)
    print()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    create_report(result, decimal_places=decimal_places)
    report = sys.stdout.getvalue()
    sys.stdout = old_stdout
    report = report.replace("_", r"\_")
    report = report.replace("+-", r"$\pm$")
    report = report.replace("(d=", "($d$=")
    report = report.replace("(delta=", r"($\delta$=")
    report = report.replace("is alpha", r"$\alpha$")
    print(report.strip())
    print()

    if len(result.rankdf) > 2:
        latex_table(result, decimal_places=decimal_places, label='tbl:%sstat_results' % prefix)
        print()

    if result.omnibus != 'wilcoxon' and result.omnibus != 'bayes' and generate_plots and result.pvalue < result.alpha:
        # only include plots if the results are significant
        plot_stats(result)
        if len(figure_path) > 0 and not figure_path.endswith("/"):
            figure_path += '/'
        figure_path = "%s%sstat_results.pdf" % (figure_path, prefix)
        plt.savefig(figure_path)

        print(r"\begin{figure}[h]")
        print(r"\includegraphics[]{%s}" % figure_path)
        if result.posthoc == 'nemenyi':
            print(r"\caption{CD diagram to visualize the results of the Nemenyi post-hoc test. The horizontal lines "
                  r"indicate that differences are not significant.}")
        elif result.posthoc == 'TukeyHSD' or result.posthoc == 'ttest':
            print(r"\caption{Confidence intervals and mean values of the populations.}")
        else:
            # fallback in case of unknown post-hoc test. should not happen
            print(r"\caption{Plot of the results}")
        print(r"\label{fig:%sstats_fig}" % prefix)
        print(r"\end{figure}")
        print()

    if complete_document:
        print(r"\end{document}")