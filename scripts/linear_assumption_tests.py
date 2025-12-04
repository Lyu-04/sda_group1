import os
import matplotlib.pyplot as plt

def plot_residuals_vs_year(yearly, metrics, metric, outdir='../plots/residuals'):
    """
    Plot residuals of a metric vs year.

    yearly : pandas.DataFrame
        DataFrame containing yearly averages for each metric.
    metrics : dict
        Dictionary with regression results: {metric: (slope, intercept, r2, p)}
    metric : str
        Metric to plot ('mag', 'len', 'wid')
    outdir : str
        Directory to save plots
    """
    slope, intercept, r2, p = metrics[metric]
    residuals = yearly[metric] - (intercept + slope * yearly['yr'])

    plt.figure(figsize=(10,6))
    plt.scatter(yearly['yr'], residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Year ({metric})')

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f'residuals_vs_year_{metric}.png')
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Residual plot saved to {outfile}")


def test_linearity_all_metrics(yearly, metrics, metric_list, outdir='../plots'):
    """
    Loop over metrics and plot residuals for each.

    yearly : pandas.DataFrame
        DataFrame containing yearly averages for each metric.
    metrics : dict
        Dictionary with regression results: {metric: (slope, intercept, r2, p)}
    metric_list : list
        List of metrics to test
    outdir : str
        Directory to save plots
    """
    for metric in metric_list:
        plot_residuals_vs_year(yearly, metrics, metric, outdir=outdir)
