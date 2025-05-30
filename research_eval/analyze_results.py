import sqlite3
import pandas as pd
import argparse
import os
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import json # For potentially parsing JSON strings if needed

# Ensure the project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from research_eval.db_utils import DB_FILE
except ImportError:
    # Fallback if db_utils isn't updated or available in a specific test environment
    DB_FILE = "experiment_results.db" 
    logging.warning("Could not import DB_FILE from research_eval.db_utils, using default 'experiment_results.db'")


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def query_results(db_file, sql_query, params=()) -> pd.DataFrame:
    """
    Connects to the SQLite database, executes the given SQL query,
    and returns the results as a Pandas DataFrame.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query(sql_query, conn, params=params)
        logging.info(f"Query executed successfully. Fetched {len(df)} rows.")
        return df
    except sqlite3.Error as e:
        logging.error(f"Error querying database {db_file}: {e}")
        return pd.DataFrame() 
    finally:
        if conn:
            conn.close()

def generate_summary_table(db_file, dataset_name, model_types=None, metric_to_optimize='final_val_f1', output_dir=".") -> pd.DataFrame:
    """
    Generates a summary table of experiment results, focusing on the best run per model type
    based on the metric_to_optimize.
    """
    logging.info(f"Generating summary table for dataset: {dataset_name}, optimizing for: {metric_to_optimize}")
    if model_types:
        logging.info(f"Filtering for model types: {model_types}")

    sql_query_all_metrics = """
    SELECT
        e.experiment_id,
        e.run_type,
        e.experiment_name,
        e.model_type,
        e.timestamp,
        e.epochs,
        e.batch_size,
        e.full_config_json, 
        m.metric_name,
        m.metric_value
    FROM experiments e
    JOIN metrics m ON e.experiment_id = m.experiment_id
    WHERE e.dataset_name = ?
    """
    params = [dataset_name]

    if model_types:
        placeholders = ','.join('?' for _ in model_types)
        sql_query_all_metrics += f" AND e.model_type IN ({placeholders})"
        params.extend(model_types)
    
    df_long = query_results(db_file, sql_query_all_metrics, params=params)

    if df_long.empty:
        logging.warning("No data found for the specified criteria for summary table.")
        return pd.DataFrame()

    try:
        index_cols = ['experiment_id', 'run_type', 'experiment_name', 'model_type', 'timestamp', 'epochs', 'batch_size', 'full_config_json']
        index_cols = [col for col in index_cols if col in df_long.columns]

        df_pivot = df_long.pivot_table(index=index_cols, 
                                       columns='metric_name', 
                                       values='metric_value').reset_index()
    except Exception as e:
        logging.error(f"Error pivoting data: {e}. DataFrame head:\n{df_long.head()}")
        # Fallback for difficult to pivot data (e.g. if metric_value has mixed types not handled by REAL)
        # Try to extract at least some common numeric metrics.
        common_numeric_metrics = ['final_val_f1', 'param_count', 'inference_latency_ms_batch', 'total_training_time_seconds', 'final_val_acc', 'final_val_loss']
        df_filtered_long = df_long[df_long['metric_name'].isin(common_numeric_metrics)]
        
        # Convert metric_value to numeric, coercing errors
        df_filtered_long['metric_value'] = pd.to_numeric(df_filtered_long['metric_value'], errors='coerce')
        df_filtered_long.dropna(subset=['metric_value'], inplace=True) # Drop rows where conversion failed
        
        if df_filtered_long.empty:
            logging.error("No valid numeric metrics found after coercion for pivoting.")
            return pd.DataFrame()

        df_pivot = df_filtered_long.pivot_table(index=index_cols, 
                                       columns='metric_name', 
                                       values='metric_value').reset_index()
        logging.warning("Initial pivot failed, retried with common numeric metrics after coercion.")


    if df_pivot.empty:
        logging.warning("Pivoted data is empty for summary table.")
        return pd.DataFrame()

    if metric_to_optimize not in df_pivot.columns:
        logging.warning(f"Metric '{metric_to_optimize}' not found in pivoted data columns: {df_pivot.columns.tolist()}. Cannot determine best runs. Returning all runs.")
        summary_df = df_pivot
    else:
        higher_is_better_metrics = ['final_val_f1', 'final_val_acc', 'best_val_f1_across_epochs'] 
        ascending_sort = metric_to_optimize not in higher_is_better_metrics

        df_pivot[metric_to_optimize] = pd.to_numeric(df_pivot[metric_to_optimize], errors='coerce')
        df_pivot.dropna(subset=[metric_to_optimize], inplace=True) # Ensure the optimization metric is valid

        if df_pivot.empty:
            logging.warning(f"Pivoted data is empty after coercing/dropping NaNs for metric '{metric_to_optimize}'.")
            return pd.DataFrame()

        # For idxmax/idxmin to work correctly, ensure the metric column is numeric
        if ascending_sort: # Lower is better
            idx = df_pivot.loc[df_pivot.groupby(['model_type'], dropna=False)[metric_to_optimize].idxmin()]
        else: # Higher is better
            idx = df_pivot.loc[df_pivot.groupby(['model_type'], dropna=False)[metric_to_optimize].idxmax()]
        summary_df = idx.copy() # Use .copy() to avoid SettingWithCopyWarning
        
        summary_df = summary_df.sort_values(by=metric_to_optimize, ascending=ascending_sort)


    cols_to_show = ['model_type', 'experiment_name', 'run_type', 'timestamp', 'epochs', 'batch_size',
                    metric_to_optimize, 'param_count', 'inference_latency_ms_batch', 
                    'total_training_time_seconds', 'experiment_id', 'full_config_json']
    
    final_summary_cols = [col for col in cols_to_show if col in summary_df.columns]
    summary_df_final = summary_df[final_summary_cols]
    
    csv_path = os.path.join(output_dir, f"{dataset_name}_summary_table_best_{metric_to_optimize.replace('_', '')}.csv")
    summary_df_final.to_csv(csv_path, index=False)
    logging.info(f"Summary table saved to {csv_path}")
    
    return summary_df_final

def plot_f1_vs_params(db_file, dataset_name, model_types=None, output_dir=".", output_image_name=None):
    logging.info(f"Generating F1 vs. Params plot for dataset: {dataset_name}")
    if model_types:
        logging.info(f"Filtering for model types: {model_types}")

    sql_query = """
    SELECT
        e.experiment_id,
        e.model_type,
        MAX(CASE WHEN m.metric_name = 'final_val_f1' THEN m.metric_value ELSE NULL END) as final_val_f1,
        MAX(CASE WHEN m.metric_name = 'param_count' THEN m.metric_value ELSE NULL END) as param_count
    FROM experiments e
    JOIN metrics m ON e.experiment_id = m.experiment_id
    WHERE e.dataset_name = ? 
      AND m.metric_name IN ('final_val_f1', 'param_count')
    """
    params = [dataset_name]

    if model_types:
        placeholders = ','.join('?' for _ in model_types)
        sql_query += f" AND e.model_type IN ({placeholders})"
        params.extend(model_types)
    
    sql_query += " GROUP BY e.experiment_id, e.model_type"
    
    plot_df = query_results(db_file, sql_query, params=params)

    if plot_df.empty or 'final_val_f1' not in plot_df.columns or 'param_count' not in plot_df.columns:
        logging.warning("Not enough data (F1 or param_count missing) to generate F1 vs. Params plot.")
        return

    plot_df['final_val_f1'] = pd.to_numeric(plot_df['final_val_f1'], errors='coerce')
    plot_df['param_count'] = pd.to_numeric(plot_df['param_count'], errors='coerce')
    plot_df.dropna(subset=['final_val_f1', 'param_count'], inplace=True)

    if plot_df.empty:
        logging.warning("Data is empty after dropping NaNs for F1 vs. Params plot.")
        return

    plt.figure(figsize=(12, 7)) # Adjusted size for better legend placement
    sns.scatterplot(data=plot_df, x='param_count', y='final_val_f1', hue='model_type', 
                    size='param_count', sizes=(50, 600), alpha=0.7, legend="auto")
    plt.title(f'F1 Score vs. Parameter Count on {dataset_name}')
    plt.xlabel('Parameter Count (Log Scale)')
    plt.ylabel('Final Validation F1 Score')
    plt.xscale('log') 
    plt.legend(title='Model Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    if not output_image_name:
        output_image_name = f"{dataset_name}_f1_vs_params.png"
    image_path = os.path.join(output_dir, output_image_name)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.savefig(image_path)
    logging.info(f"F1 vs. Params plot saved to {image_path}")
    plt.close()


def plot_f1_vs_latency(db_file, dataset_name, model_types=None, output_dir=".", output_image_name=None):
    logging.info(f"Generating F1 vs. Latency plot for dataset: {dataset_name}")
    if model_types:
        logging.info(f"Filtering for model types: {model_types}")

    sql_query = """
    SELECT
        e.experiment_id,
        e.model_type,
        MAX(CASE WHEN m.metric_name = 'final_val_f1' THEN m.metric_value ELSE NULL END) as final_val_f1,
        MAX(CASE WHEN m.metric_name = 'inference_latency_ms_batch' THEN m.metric_value ELSE NULL END) as inference_latency_ms_batch
    FROM experiments e
    JOIN metrics m ON e.experiment_id = m.experiment_id
    WHERE e.dataset_name = ?
      AND m.metric_name IN ('final_val_f1', 'inference_latency_ms_batch')
    """
    params = [dataset_name]

    if model_types:
        placeholders = ','.join('?' for _ in model_types)
        sql_query += f" AND e.model_type IN ({placeholders})"
        params.extend(model_types)
        
    sql_query += " GROUP BY e.experiment_id, e.model_type"

    plot_df = query_results(db_file, sql_query, params=params)

    if plot_df.empty or 'final_val_f1' not in plot_df.columns or 'inference_latency_ms_batch' not in plot_df.columns:
        logging.warning("Not enough data (F1 or latency missing) to generate F1 vs. Latency plot.")
        return
        
    plot_df['final_val_f1'] = pd.to_numeric(plot_df['final_val_f1'], errors='coerce')
    plot_df['inference_latency_ms_batch'] = pd.to_numeric(plot_df['inference_latency_ms_batch'], errors='coerce')
    plot_df.dropna(subset=['final_val_f1', 'inference_latency_ms_batch'], inplace=True)

    if plot_df.empty:
        logging.warning("Data is empty after dropping NaNs for F1 vs. Latency plot.")
        return

    plt.figure(figsize=(12, 7)) # Adjusted size
    sns.scatterplot(data=plot_df, x='inference_latency_ms_batch', y='final_val_f1', hue='model_type', 
                    size='inference_latency_ms_batch', sizes=(50, 600), alpha=0.7, legend="auto")
    plt.title(f'F1 Score vs. Inference Latency on {dataset_name}')
    plt.xlabel('Inference Latency (ms/batch) (Log Scale)')
    plt.ylabel('Final Validation F1 Score')
    plt.xscale('log') 
    plt.legend(title='Model Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    if not output_image_name:
        output_image_name = f"{dataset_name}_f1_vs_latency.png"
    image_path = os.path.join(output_dir, output_image_name)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
    plt.savefig(image_path)
    logging.info(f"F1 vs. Latency plot saved to {image_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze experiment results from SQLite database.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to analyze (e.g., 'parity', 'ag_news').")
    parser.add_argument('--model_types', type=str, help="Comma-separated list of model types to include (e.g., 'dense,moe'). Optional.")
    parser.add_argument('--output_dir', type=str, default="analysis_reports", help="Directory to save generated reports and plots.")
    parser.add_argument('--metric_to_optimize', type=str, default='final_val_f1', help="Metric to use for selecting best runs in summary table.")
    
    parser.add_argument('--run_summary_table', action='store_true', help="Generate and save the summary table.")
    parser.add_argument('--run_f1_vs_params_plot', action='store_true', help="Generate and save the F1 vs. Params plot.")
    parser.add_argument('--run_f1_vs_latency_plot', action='store_true', help="Generate and save the F1 vs. Latency plot.")
    parser.add_argument('--run_all', action='store_true', help="Run all analysis types.")

    args = parser.parse_args()

    if not os.path.exists(DB_FILE):
        logging.error(f"Database file {DB_FILE} not found. Please run init_db.py or ensure the path is correct.")
        sys.exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    model_types_list = [mt.strip() for mt in args.model_types.split(',')] if args.model_types else None

    if args.run_summary_table or args.run_all:
        logging.info(f"\n--- Generating Summary Table for {args.dataset_name} ---")
        summary_table = generate_summary_table(DB_FILE, args.dataset_name, model_types_list, args.metric_to_optimize, args.output_dir)
        if not summary_table.empty:
            print("\nSummary Table:")
            print(summary_table.to_string())
        else:
            print("No data for summary table (or function not fully implemented yet).")

    if args.run_f1_vs_params_plot or args.run_all:
        logging.info(f"\n--- Generating F1 vs. Params Plot for {args.dataset_name} ---")
        plot_f1_vs_params(DB_FILE, args.dataset_name, model_types_list, args.output_dir)

    if args.run_f1_vs_latency_plot or args.run_all:
        logging.info(f"\n--- Generating F1 vs. Latency Plot for {args.dataset_name} ---")
        plot_f1_vs_latency(DB_FILE, args.dataset_name, model_types_list, args.output_dir)

    if not (args.run_summary_table or args.run_f1_vs_params_plot or args.run_f1_vs_latency_plot or args.run_all):
        logging.info("No analysis type specified. Use --run_summary_table, --run_f1_vs_params_plot, --run_f1_vs_latency_plot, or --run_all.")
        parser.print_help()
