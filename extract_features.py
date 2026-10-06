"""Extract features from raw dataset."""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


from src import features

def extract_all_features(input_csv: Path, output_csv: Path) -> None:
    print(f"Reading {input_csv}...")
    # The raw CSV has no headers, assuming URL and Label
    df = pd.read_csv(input_csv, header=None, names=["url", "label"])
    
    # We will compute the features for each URL
    features_list = []
    
    # List of feature functions
    feature_funcs = [
        ("url_length", features.url_length),
        ("domain_length", features.domain_length),
        ("number_of_dots", features.number_of_dots),
        ("number_of_subdomains", features.number_of_subdomains),
        ("number_of_hyphens", features.number_of_hyphens),
        ("number_of_at_symbols", features.number_of_at_symbols),
        ("number_of_digits", features.number_of_digits),
        ("has_ip_address", features.has_ip_address),
        ("is_https", features.is_https),
        ("suspicious_keyword_count", features.suspicious_keyword_count),
        ("has_punycode", features.has_punycode),
        # Note: some features like domain_age, dns_record_exists, number_of_external_links, 
        # and number_of_password_fields do external network requests, which might be slow.
        ("url_path_entropy", features.url_path_entropy)
    ]
    
    print(f"Extracting features for {len(df)} URLs...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row["url"]
        row_features = {"url": url, "label": row["label"]}
        for name, func in feature_funcs:
            try:
                row_features[name] = func(url)
            except Exception:
                row_features[name] = 0 # Default/fallback
        features_list.append(row_features)
        
    out_df = pd.DataFrame(features_list)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from URL dataset.")
    parser.add_argument("--input-csv", type=Path, default=Path("full_dataset.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/features.csv"))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_all_features(args.input_csv, args.output_csv)

