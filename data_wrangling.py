"""
data_wrangling.py

Loads raw World Bank GDP per capita and Life Expectancy CSVs,
cleans and reshapes them, merges into a single DataFrame,
handles missing values, and saves processed CSV.
"""

import pandas as pd
import os


def load_indicator(filepath, value_name):
    """
    Load a World Bank indicator CSV, drop metadata columns and unnamed blanks, and melt into long format.

    Args:
        filepath (str): Path to the raw CSV file (skiprows=4).
        value_name (str): Name for the value column after melting.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns [Country Name, Country Code, Year, value_name].
    """
    # Read CSV, skipping the first four metadata rows
    df = pd.read_csv(filepath, skiprows=4)
    # Drop redundant and blank columns: indicator metadata and unnamed extras
    drop_cols = [col for col in df.columns if col in ["Indicator Name", "Indicator Code"] or col.startswith("Unnamed")]
    df = df.drop(columns=drop_cols)
    # Identify year columns (numeric column names)
    year_cols = [col for col in df.columns if col.isdigit()]
    # Melt from wide (years as columns) to long format
    df_long = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name=value_name
    )
    # Convert Year to integer type
    df_long["Year"] = df_long["Year"].astype(int)
    return df_long


def main():
    # Define raw and processed directories
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # File paths (ensure your raw CSVs are placed here)
    gdp_file = "data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_19346.csv"
    life_file = "data/API_SP.DYN.LE00.IN_DS2_en_csv_v2_19383.csv"

    # Load each indicator into long-form DataFrames
    gdp_df = load_indicator(gdp_file, "GDP_per_capita")
    life_df = load_indicator(life_file, "Life_Expectancy")

    # Merge indicators on Country Name, Country Code, and Year
    df = pd.merge(
        gdp_df,
        life_df,
        on=["Country Name", "Country Code", "Year"],
        how="inner"
    )

    # Handle missing values: fill with median of each column
    df["GDP_per_capita"] = df["GDP_per_capita"].fillna(df["GDP_per_capita"].median())
    df["Life_Expectancy"] = df["Life_Expectancy"].fillna(df["Life_Expectancy"].median())

    # Save the cleaned, merged dataset
    output_file = os.path.join(processed_dir, "country_health_econ.csv")
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


if __name__ == '__main__':
    main()
