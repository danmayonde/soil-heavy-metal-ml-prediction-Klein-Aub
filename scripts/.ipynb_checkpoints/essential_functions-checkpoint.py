"""
EssentialFunctions Module

This module contains essential functions for data exploration and analysis, and for visualisation.
Functions will be added here as needed for dataset exploration, analysis and plotting tasks.

Author: Dan Manengo Mayonde
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, chi2
from scipy import stats
import shap
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def descriptive_statistics(df: pd.DataFrame, output_filename: str, save: bool = False):
    """
    Compute descriptive statistics for a given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        output_filename (str): The name of the CSV file to save the statistics (if save=True).
        save (bool): Whether to save the statistics DataFrame as a CSV file. Default is False.

    Returns:
        stats (pd.DataFrame): DataFrame containing the descriptive statistics for numeric columns.
        freq_stats (dict): Dictionary with frequency statistics for categorical columns.
    """
    # Numeric stats
    stats = pd.DataFrame(index=df.select_dtypes(include=[np.number]).columns)
    stats['Min'] = df[stats.index].min()
    stats['Max'] = df[stats.index].max()
    stats['Mean'] = df[stats.index].mean()
    stats['SD'] = df[stats.index].std()
    stats['Median'] = df[stats.index].median()
    stats['CV'] = stats['SD'] / stats['Mean']
    stats['Skewness'] = df[stats.index].apply(lambda x: skew(x.dropna(), bias=False), axis=0)
    stats['Kurtosis'] = df[stats.index].apply(lambda x: kurtosis(x.dropna()), axis=0)
    stats['z_skewness'] = df[stats.index].apply(lambda x: (skew(x.dropna(), bias=False)/np.sqrt(6/len(x.dropna()))), axis=0)
    stats['Q1'] = df[stats.index].quantile(0.25)
    stats['Q3'] = df[stats.index].quantile(0.75)
    stats['IQR'] = df[stats.index].quantile(0.75) - df[stats.index].quantile(0.25)
    
    # Categorical stats
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    freq_stats = {}
    for col in cat_cols:
        freq_stats[col] = df[col].value_counts(dropna=False)

    if save:
        output_dir = os.path.join('../Data', 'Description_Stats')
        output_path = os.path.join(output_dir, output_filename)
        stats.to_csv(output_path)
        # Optionally, save categorical stats as well
        if freq_stats:
            for col, freq in freq_stats.items():
                freq.to_csv(os.path.join(output_dir, f"{col}_frequency.csv"))

    return stats, freq_stats


def pearson_correlation_analysis(
    df: pd.DataFrame,
    output_filename: str = "correlation_table.csv",
    save: bool = False,
    mask: bool = True,
    plot_filename: str = "correlation_matrix.png",
    plot: bool = False
):
    """
    Compute the Pearson correlation matrix for the features in the dataset.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        output_filename (str): The name of the CSV file to save the correlation matrix (if save=True). Default is 'correlation_table.csv'.
        save (bool): Whether to save the correlation matrix as a CSV file. Default is False.
        plot_filename (str): The name of the image file to save the correlation heatmap (if plot=True). Default is 'correlation_matrix.png'.
        plot (bool): Whether to plot and save the correlation matrix as a heatmap image. Default is False.

    Returns:
        pd.DataFrame: The Pearson correlation matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found in df to compute Spearman correlation.")
        
    corr_matrix = numeric_df.corr(method='pearson')

    if save:
        output_path = os.path.join('../correlations', output_filename)
        corr_matrix.to_csv(output_path)

    if plot:        
        if mask:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            plt.figure(figsize=(16, 10))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f')
        else:
            plt.figure(figsize=(16, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')

        plt.title('Pearson Correlation Matrix')
        plt.tight_layout()
        plot_path = os.path.join('../correlations', plot_filename)
        plt.savefig(plot_path)
        # plt.close()
        plt.show()

    return corr_matrix


def spearman_correlation_analysis(
    df: pd.DataFrame,
    output_filename: str = "spearman_correlation_table.csv",
    save: bool = False,
    mask: bool =True,
    plot_filename: str = "spearman_correlation_matrix.png",
    plot: bool  = False
) -> pd.DataFrame:
    """
    Compute the Spearman correlation matrix for the features in the dataset.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        output_filename (str): The name of the CSV file to save the correlation matrix (if save=True).
            The file will be saved under the 'Correlations' directory.
        save (bool): Whether to save the correlation matrix as a CSV file. Default is False.
        plot_filname (str): The name of the image file to save the correlation heatmap (if plot=True).
            The file will be saved under the 'Correlations' directory. Default is 'correlation_matrix.png'.
        plot (bool): Whether to plot and save the correlation matrix as a heatmap image. Default is False.

    Returns:
        pd.DataFrame: The Spearman correlation matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found in df to compute Spearman correlation.")

    corr_matrix = numeric_df.corr(method="spearman")

    if save:
        output_path = os.path.join("../correlations", output_filename)
        corr_matrix.to_csv(output_path)

    if plot:
        if mask:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            plt.figure(figsize=(16, 10))
            ax = sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap=sns.diverging_palette(220, 15, as_cmap=True),  # deep blue -> warm red
                annot=True,
                fmt=".2f",
            )
        else:
            plt.figure(figsize=(16, 10))
            ax = sns.heatmap(
                corr_matrix,
                cmap=sns.diverging_palette(220, 15, as_cmap=True),  # deep blue -> warm red
                annot=True,
                fmt=".2f",
            )
            
        ax.set_title("Spearman Correlation Matrix", pad=14, fontsize=14, weight="bold")
        ax.tick_params(axis="x", labelrotation=45, labelsize=10)
        ax.tick_params(axis="y", labelrotation=0, labelsize=10)
        plt.tight_layout()

        plot_path = os.path.join("../correlations", plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()

    return corr_matrix


def circular_descriptive_statistics(df: pd.DataFrame,
                                     output_filename: str = '', 
                                     save: bool = False):
    """
    Compute circular descriptive statistics for all numeric columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing circular data in degrees
    output_filename : str
        Name of the CSV file to save the statistics (required if save=True)
    save : bool, default=False
        If True, save the statistics DataFrame as a CSV file
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing circular descriptive statistics:
        - Circular Mean (degrees)
        - Circular Variance
        - Circular Standard Deviation (degrees)
        - Resultant Vector Length
        - Rayleigh Test Statistic
        - Rayleigh Test P-value
    """
    # Convert degrees to radians for computation
    df_radians = df.select_dtypes(include=[np.number]).apply(lambda x: np.deg2rad(x))
    
    # Initialize results dictionary
    results = {
        'Variable': [],
        'Circular_Mean_Degrees': [],
        'Circular_Variance': [],
        'Circular_Std_Degrees': [],
        'Resultant_Vector_Length': [],
        'Rayleigh_Statistic': [],
        'Rayleigh_Pvalue': []
    }
    
    # Compute statistics for each numeric column
    for col in df_radians.columns:
        # Remove NaN values
        data = df_radians[col].dropna()
        
        if len(data) == 0:
            continue
        
        # Circular mean (in radians)
        circ_mean_rad = stats.circmean(data, high=2*np.pi, low=0)
        
        # Circular variance
        circ_var = stats.circvar(data, high=2*np.pi, low=0)
        
        # Circular standard deviation (in radians)
        circ_std_rad = stats.circstd(data, high=2*np.pi, low=0)
        
        # Resultant vector length (R)
        # R = sqrt((sum(cos(θ)))^2 + (sum(sin(θ)))^2) / n
        cos_sum = np.sum(np.cos(data))
        sin_sum = np.sum(np.sin(data))
        R = np.sqrt(cos_sum**2 + sin_sum**2) / len(data)
        
        # Rayleigh test for uniformity
        # The test statistic Z = n*R^2 follows a chi-square distribution with 2 df under H0
        rayleigh_stat = len(data) * R**2
        rayleigh_pvalue = 1 - chi2.cdf(rayleigh_stat, df=2)
        
        # Convert mean and std back to degrees
        circ_mean_deg = np.rad2deg(circ_mean_rad)
        circ_std_deg = np.rad2deg(circ_std_rad)
        
        # Store results
        results['Variable'].append(col)
        results['Circular_Mean_Degrees'].append(circ_mean_deg)
        results['Circular_Variance'].append(circ_var)
        results['Circular_Std_Degrees'].append(circ_std_deg)
        results['Resultant_Vector_Length'].append(R)
        results['Rayleigh_Statistic'].append(rayleigh_stat)
        results['Rayleigh_Pvalue'].append(rayleigh_pvalue)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV if requested
    if save:
        if not output_filename:
            raise ValueError("output_filename must be provided when save=True")
        output_dir = os.path.join('../Data', 'Description_Stats')
        output_path = os.path.join(output_dir, output_filename)
        results_df.to_csv(output_path)
    
    return results_df

def slope_aspect_direction(df, output_filename='', save=False):
    """
    Add a 'Direction' column to a DataFrame based on angle values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with a single column containing angle values (in degrees)
    save : bool, default=False
        If True, save the DataFrame with the new Direction column to a CSV file
    output_path : str
        Path/filename for saving the CSV file (required if save=True)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with the original column and a new 'Direction' column
    """
    # Make a copy to avoid modifying the original DataFrame
    aspect_direction_df = df.copy()
    
    # Get the single column name
    col_name = df.columns[0]
    
    # Create Direction column based on angle ranges
    def assign_direction(angle):
        if (angle >= 0 and angle <= 45) or (angle > 315 and angle < 360):
            return 'North'
        elif angle > 45 and angle <= 135:
            return 'East'
        elif angle > 135 and angle <= 225:
            return 'South'
        elif angle > 225 and angle <= 315:
            return 'West'
        else:
            return None  # For values outside 0-360 range
    
    aspect_direction_df['Direction'] = aspect_direction_df[col_name].apply(assign_direction)
    
    # Save to CSV if requested
    if save:
        output_dir = os.path.join('../Data', 'Description_Stats')
        output_path = os.path.join(output_dir, output_filename)
        aspect_direction_df.to_csv(output_path)
    
    return aspect_direction_df

def scatter_comparison_train_test_plot(y_pred, 
                                       y_test, 
                                       y_train_pred, 
                                       y_train, 
                                       x_label: str = 'Measured Cu Conc. (mg/kg)', 
                                       y_label: str = "Predicted Cu Conc. (mg/kg)",
                                       title: str = '',
                                       file_name: str ='scatter_plot.png'
                                      ):
    
    # True and predicted for test set
    y_true_test = y_test
    y_pred_test = y_pred
    
    # True and predicted for train set
    y_true_train = y_train
    y_pred_train = y_train_pred
    
    x_label = x_label
    y_label = y_label
    title = title
    # -----------------------------------
    
    def _clean_pair(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        m = ~np.isnan(y_true) & ~np.isnan(y_pred)
        return y_true[m], y_pred[m]
    
    yt_test, yp_test = _clean_pair(y_true_test, y_pred_test)
    yt_train, yp_train = _clean_pair(y_true_train, y_pred_train)
    
    all_true = np.concatenate([yt_test, yt_train])
    all_pred = np.concatenate([yp_test, yp_train])
    
    data_min = min(all_true.min(), all_pred.min())
    data_max = max(all_true.max(), all_pred.max())
    pad = 0.02 * (data_max - data_min) if data_max > data_min else 1.0
    x_min, x_max = data_min - pad, data_max + pad
    
    # R² on test set (most commonly reported)
    ss_res = np.sum((yt_test - yp_test) ** 2)
    ss_tot = np.sum((yt_test - np.mean(yt_test)) ** 2)
    r2_test = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=140)
    
    # Colors
    color_test = "#2A9D8F"   
    color_train = "#E76F51"  
    identity_color = "#264653"
    
    # Test scatter
    ax.scatter(
        yt_test,
        yp_test,
        s=22,
        alpha=0.7,
        color=color_test,
        edgecolor="none",
        label="Test",
    )
    
    # Train scatter
    ax.scatter(
        yt_train,
        yp_train,
        s=20,
        alpha=0.45,
        color=color_train,
        edgecolor="none",
        label="Train",
    )
    
    # Identity line
    ax.plot(
        [x_min, x_max],
        [x_min, x_max],
        color=identity_color,
        linestyle="--",
        linewidth=1.6,
        zorder=3,
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    
    # Annotate R² (test)
    ax.text(
        0.04,
        0.96,
        f"$R^2_{{test}} = {r2_test:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
    )
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#e0e0e0", fontsize=10)

    output_path = "../feature_importance"
    plt.tight_layout()
    plt.savefig(f'{output_path}/{file_name}')
    plt.show()

def feature_importance_plot(X_test,
                            shap_values_Cu,
                            shap_values_Zn,
                            shap_values_Pb,
                            plot_name = "shap_feature_importance_summary.png"):
    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(16, 14))

    # Set the output directory
    output_path = "../feature_importance"

    # Feature names
    feature_names = X_test.columns.tolist() 

    plt.sca(ax0)
    shap.summary_plot(
        shap_values_Cu,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        color="#4A90A4",
        plot_size=(16, 14)
)

    plt.sca(ax1)
    shap.summary_plot(
        shap_values_Cu,
        X_test,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        plot_size=(16, 14)
        # cmap="seismic",
    )

    


    plt.sca(ax2)
    shap.summary_plot(
        shap_values_Zn,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        color="#4A90A4",
        plot_size=(16, 14)
)

    plt.sca(ax3)
    shap.summary_plot(
        shap_values_Zn,
        X_test,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        plot_size=(16, 14)
        # cmap="seismic",
    )

    plt.sca(ax4)
    shap.summary_plot(
        shap_values_Pb,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        color="#4A90A4",
        plot_size=(16, 14)
)
    
    plt.sca(ax5)
    shap.summary_plot(
        shap_values_Pb,
        X_test,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        plot_size=(16, 14)
        # cmap="seismic",
    )

    ax0.set_title("(a)",
                fontsize=14,
                fontweight="bold",
                fontstyle="italic",
                loc="left")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.tick_params(axis="both", labelsize=10)

    ax1.set_title("(b)",
                fontsize=14,
                fontweight="bold",
                fontstyle="italic",
                loc="left")
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.tick_params(axis="both", labelsize=10)

    ax2.set_title("(c)",
                fontsize=14,
                fontweight="bold",
                fontstyle="italic",
                loc="left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", labelsize=10)

    ax3.set_title("(d)",
                fontsize=14,
                fontweight="bold",
                fontstyle="italic",
                loc="left")
    ax3.spines["top"].set_visible(True)
    ax3.spines["right"].set_visible(True)
    ax3.tick_params(axis="both", labelsize=10)


    ax4.set_title("(e)",
                fontsize=14,
                fontweight="bold",
                fontstyle="italic",
                loc="left")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.tick_params(axis="both", labelsize=10)

    ax5.set_title("(f)",
                fontsize=14,
                fontweight="bold",
                fontstyle="italic",
                loc="left")
    ax5.spines["top"].set_visible(True)
    ax5.spines["right"].set_visible(True)
    ax5.tick_params(axis="both", labelsize=10)


    plt.tight_layout()
    plt.savefig(f"{output_path}/{plot_name}", dpi=300, bbox_inches="tight")
    plt.show()