#!/usr/bin/env python3
"""
Focused analyzer for processed sepsis data files:
- deep_static.csv
- deep_vitals_4H_organized.csv  
- deep_labs_4H_organized.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_static_variables(file_path, valid_records=None):
    """
    Analyze static variables from deep_static.csv
    Handles post-processed one-hot encoded data correctly
    Applies exact record filter if provided (PID, AID, infection_id)
    """
    print("📊 Analyzing Static Variables...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    try:
        # Read the static file
        df = pd.read_csv(file_path)
        original_records = len(df)
        print(f"  📈 Original records: {original_records:,}")
        
        # Apply exact record filter if provided
        if valid_records is not None:
            # Create a set of valid (PID, AID, infection_id) tuples for fast lookup
            valid_tuples = set(valid_records[['PID', 'AID', 'infection_id']].itertuples(index=False, name=None))
            
            # Filter dataframe to only include valid records
            mask = df[['PID', 'AID', 'infection_id']].apply(tuple, axis=1).isin(valid_tuples)
            df = df[mask]
            
            filtered_records = len(df)
            print(f"  🔍 After exact filtering: {filtered_records:,} records ({filtered_records/original_records*100:.1f}%)")
        
        max_records = len(df)
        print(f"  📋 Total variables: {len(df.columns)}")
        
        results = []
        
        # Skip ID columns
        id_columns = ['PID', 'AID', 'infection_id']
        variables = [col for col in df.columns if col not in id_columns]
        
        # Handle grouped one-hot encoded variables
        processed_groups = set()
        
        for var in variables:
            # Skip if already processed as part of a group
            if var in processed_groups:
                continue
            
            # Group gender variables
            if var.startswith('gender_'):
                if 'gender' not in [r['Variable'] for r in results]:  # Only add once
                    unknown_count = df['gender_U'].sum() if 'gender_U' in df.columns else 0
                    actual_records = max_records - unknown_count
                    availability_rate = (actual_records / max_records) * 100
                    
                    result = {
                        'Data_modality': 'Static',
                        'Variable': 'Gender',
                        'Data_type': 'Categorical',
                        'Category': '2',  # F, M
                        'Total_records': int(actual_records),
                        'Availability_rate': f"{availability_rate:.2f}%"
                    }
                    results.append(result)
                
                # Mark all gender variables as processed
                for g_var in [v for v in variables if v.startswith('gender_')]:
                    processed_groups.add(g_var)
                continue
            
            # Group race variables  
            if var.startswith('race_'):
                if 'race' not in [r['Variable'] for r in results]:  # Only add once
                    unknown_count = df['race_ '].sum() if 'race_ ' in df.columns else 0
                    actual_records = max_records - unknown_count
                    availability_rate = (actual_records / max_records) * 100
                    
                    result = {
                        'Data_modality': 'Static',
                        'Variable': 'Race',
                        'Data_type': 'Categorical', 
                        'Category': '5',  # 5 race categories
                        'Total_records': int(actual_records),
                        'Availability_rate': f"{availability_rate:.2f}%"
                    }
                    results.append(result)
                
                # Mark all race variables as processed
                for r_var in [v for v in variables if v.startswith('race_')]:
                    processed_groups.add(r_var)
                continue
            
            # Group hospital variables
            if var.startswith('hospital_id_'):
                if 'hospital' not in [r['Variable'] for r in results]:  # Only add once
                    # Hospitals have no unknowns
                    actual_records = max_records
                    availability_rate = 100.0
                    
                    result = {
                        'Data_modality': 'Static',
                        'Variable': 'Hospital',
                        'Data_type': 'Categorical',
                        'Category': '10',  # 10 different hospitals
                        'Total_records': int(actual_records),
                        'Availability_rate': f"{availability_rate:.2f}%"
                    }
                    results.append(result)
                
                # Mark all hospital variables as processed
                for h_var in [v for v in variables if v.startswith('hospital_id_')]:
                    processed_groups.add(h_var)
                continue
            
            # Handle other individual variables normally
            var_type, category = classify_static_variable(var, df[var])
            
            # Calculate availability normally
            missing_count = df[var].isna().sum()
            actual_records = max_records - missing_count
            availability_rate = (actual_records / max_records) * 100
            
            # Clean up variable name for display
            display_name = clean_static_variable_name(var)
            
            result = {
                'Data_modality': 'Static',
                'Variable': display_name,
                'Data_type': var_type,
                'Category': category,
                'Total_records': int(actual_records),
                'Availability_rate': f"{availability_rate:.2f}%"
            }
            results.append(result)
        
        print(f"  ✅ Processed {len(results)} static variables")
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []

def analyze_longitudinal_variables(file_path, modality_name, valid_records=None):
    """
    Analyze longitudinal variables from vitals or labs files
    Calculate EXACT availability rates by counting actual non-NaN records
    Applies exact record filter if provided (PID, AID, infection_id)
    """
    print(f"📊 Analyzing {modality_name} Variables...")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    try:
        # Get column names first
        header_df = pd.read_csv(file_path, nrows=0)
        skip_columns = ['PID', 'AID', 'infection_id', 'time']
        variables = [col for col in header_df.columns if col not in skip_columns]
        
        print(f"  📋 Analyzing {len(variables)} {modality_name.lower()} variables...")
        print("  🔢 Calculating exact availability rates...")
        
        # Initialize counters for each variable
        non_nan_counts = {var: 0 for var in variables}
        total_records = 0
        processed_records = 0
        
        # Create set of valid record tuples for fast lookup
        valid_tuples = None
        if valid_records is not None:
            valid_tuples = set(valid_records[['PID', 'AID', 'infection_id']].itertuples(index=False, name=None))
        
        # Process file in chunks to count exact non-NaN values with exact record filter
        chunk_size = 50000
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Apply exact record filter if provided
            if valid_tuples is not None:
                mask = chunk[['PID', 'AID', 'infection_id']].apply(tuple, axis=1).isin(valid_tuples)
                chunk = chunk[mask]
            
            # Count records after filtering
            total_records += len(chunk)
            
            # Count non-NaN values for each variable
            for var in variables:
                if var in chunk.columns:
                    non_nan_counts[var] += chunk[var].notna().sum()
            
            processed_records += len(chunk)
            if processed_records % 100000 == 0:  # Progress every 100k records
                print(f"    Processed {processed_records:,} filtered records...")
        
        print(f"  📈 Total filtered records: {total_records:,}")
        print(f"  ✅ Finished processing all filtered records")
        
        # Generate results with exact counts
        results = []
        
        # Get a small sample for variable classification only
        sample_df = pd.read_csv(file_path, nrows=1000)
        
        for var in variables:
            # Classify variable using sample
            var_type, category = classify_longitudinal_variable(var, sample_df[var])
            
            # Use exact counts
            actual_records_with_data = non_nan_counts[var]
            availability_rate = (actual_records_with_data / total_records) * 100
            
            # Clean up variable name for display
            display_name = clean_variable_name(var)
            
            result = {
                'Data_modality': f'Longitudinal ({modality_name})',
                'Variable': display_name,
                'Data_type': var_type,
                'Category': category,
                'Total_records': actual_records_with_data,
                'Availability_rate': f"{availability_rate:.2f}%"
            }
            results.append(result)
        
        print(f"  ✅ Processed {len(results)} {modality_name.lower()} variables with exact counts")
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []

def classify_static_variable(var_name, data):
    """
    Classify static variables based on name patterns and data
    """
    var_lower = var_name.lower()
    
    # Remove NaN values for analysis
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return "Unknown", ""
    
    unique_values = clean_data.unique()
    unique_count = len(unique_values)
    
    # Age (continuous)
    if 'age' in var_lower:
        return "Continuous", ""
    
    # Binary variables (common pattern in medical data)
    if unique_count == 2 and set(unique_values) <= {0, 1, 0.0, 1.0}:
        return "Binary", "2"
    
    # These are now handled in the main function as grouped variables
    # Gender categories (one-hot encoded)
    if 'gender' in var_lower:
        return "Categorical", "4"  # F, M, U, Unknown - total 4 categories
    
    # Hospital ID (one-hot encoded) 
    if 'hospital' in var_lower:
        return "Categorical", "10"  # 10 different hospitals
    
    # Race categories (one-hot encoded)
    if 'race' in var_lower:
        return "Categorical", "5"  # 5 race categories total
    
    # Time-related continuous
    if any(term in var_lower for term in ['time', 'days', 'hours']):
        return "Continuous", ""
    
    # Specific static variable classifications
    static_binary_vars = [
        'mechanical_ventilation', 'pneumonia_acquired', 'pneumonia_community', 
        'readmission', 'resistance_history', 'vasop_history'
    ]
    
    if var_name in static_binary_vars:
        return "Binary", "2"
    
    # Binary flags (0/1 pattern)
    if unique_count == 2 and all(val in [0, 1, 0.0, 1.0] for val in unique_values):
        return "Binary", "2"
    
    # Small number of categories
    if unique_count <= 10:
        return "Categorical", str(unique_count)
    
    # Default to continuous for static numeric data
    return "Continuous", ""

def classify_longitudinal_variable(var_name, data):
    """
    Classify longitudinal variables (vitals/labs)
    """
    var_lower = var_name.lower()
    
    # Remove NaN values
    clean_data = data.dropna()
    if len(clean_data) == 0:
        return "Continuous", ""
    
    # Most vital signs and lab values are continuous measurements
    if any(term in var_lower for term in ['vital_', 'lab_']):
        return "Continuous", ""
    
    # Check if it's actually categorical (rare for vitals/labs)
    unique_count = len(clean_data.unique())
    if unique_count <= 5 and len(clean_data) > 100:
        return "Categorical", str(unique_count)
    
    return "Continuous", ""

def clean_variable_name(var_name):
    """
    Clean up variable names for better display
    """
    # Remove prefixes
    name = var_name.replace('vital_', '').replace('lab_', '')
    
    # Capitalize first letter of each word
    name = ' '.join(word.capitalize() for word in name.replace('_', ' ').split())
    
    # Handle common medical abbreviations
    replacements = {
        'Bmi': 'BMI',
        'Ph': 'pH',
        'Co2': 'CO2',
        'O2': 'O2',
        'Ecg': 'ECG',
        'Bp': 'BP'
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name

def clean_static_variable_name(var_name):
    """
    Clean up static variable names for better display
    """
    # Handle specific static variable names
    name_map = {
        'age_yrs': 'Age',
        'mechanical_ventilation': 'Intubation',
        'pneumonia_acquired': 'Hospital-acquired pneumonia',
        'pneumonia_community': 'Community-acquired pneumonia', 
        'readmission': 'History of hospitalization',
        'resistance_history': 'History of resistance',
        'time_since_admission': 'Time since admission',
        'vasop_history': 'Vasopressors'
    }
    
    if var_name in name_map:
        return name_map[var_name]
    
    # Default: capitalize and replace underscores
    return ' '.join(word.capitalize() for word in var_name.replace('_', ' ').split())

def count_records_fast(file_path):
    """
    Fast record counting
    """
    try:
        with open(file_path, 'rb') as f:
            lines = sum(1 for _ in f) - 1  # Subtract header
        return lines
    except:
        return 0

def get_filtered_aids_and_records(base_path, note_type="hpi", use_unlabeled=False):
    """
    Get filtered AIDs and record indices that match dataset_2.py filtering logic exactly:
    1. Filter by AID availability in notes
    2. Exclude unknown labels (UN=True) if use_unlabeled=False
    """
    try:
        # Load notes data
        notes_path = base_path / "deep_notes.csv"
        if not notes_path.exists():
            print(f"❌ Notes file not found: {notes_path}")
            return None, None
        
        # Load labels data
        labels_path = base_path / "df_label_full.csv"
        if not labels_path.exists():
            print(f"❌ Labels file not found: {labels_path}")
            return None, None
        
        print(f"📝 Loading data for filtering (matching dataset_2.py logic)...")
        
        # Load data
        notes = pd.read_csv(notes_path)
        labels_raw = pd.read_csv(labels_path)
        
        print(f"  📊 Original labels: {len(labels_raw):,}")
        
        # Check if note_type column exists
        if note_type not in notes.columns:
            print(f"❌ Note type '{note_type}' not found in notes data")
            return None, None
        
        # Step 1: Filter by AID availability in notes (line 270 in dataset_2.py)
        valid_notes = notes[notes[note_type].notna()]
        valid_aids = set(valid_notes['AID'].unique())
        labels_raw = labels_raw[labels_raw["AID"].isin(valid_aids)]
        
        print(f"  🔍 After note filter: {len(labels_raw):,} records")
        
        # Step 2: Exclude unknown labels if use_unlabeled=False (line 279 in dataset_2.py)
        if not use_unlabeled:
            labels_raw = labels_raw[~labels_raw.UN]
            print(f"  🚫 After excluding UN labels: {len(labels_raw):,} records")
        
        # Get final valid AIDs and record identifiers
        final_valid_aids = set(labels_raw['AID'].unique())
        valid_records = labels_raw[['PID', 'AID', 'infection_id']]
        
        print(f"  ✅ Final AIDs with valid data: {len(final_valid_aids)}")
        print(f"  ✅ Final filtered records: {len(valid_records):,}")
        
        return final_valid_aids, valid_records
        
    except Exception as e:
        print(f"❌ Error in filtering: {e}")
        return None, None

def main():
    """
    Main analysis function for processed sepsis data
    Applies note availability filter matching the dataset logic
    """
    print("🏥 Sepsis Processed Data Analysis (with Note Filter)")
    print("=" * 60)
    
    # Define file paths  
    base_path = Path("/Users/ericliu/Desktop/Proj_sepsis/cohort_3_new/data_combined")
    
    # Get filtered AIDs and records (matching dataset_2.py logic exactly)
    note_type = "hpi"  # From config: note_type: "hpi"  
    use_unlabeled = False  # From config: use_unlabeled defaults to False
    valid_aids, valid_records = get_filtered_aids_and_records(base_path, note_type, use_unlabeled)
    
    if valid_aids is None or valid_records is None:
        print("❌ Cannot proceed without proper filtering")
        return None
    
    print(f"\n🔍 Filters applied:")
    print(f"  1. AID must have {note_type} notes available")
    print(f"  2. Exclude unknown labels (UN=True)" + (f" [DISABLED]" if use_unlabeled else ""))
    print("-" * 60)
    
    files_to_analyze = {
        'static': (base_path / 'deep_static.csv', 'Static'),
        'vitals': (base_path / 'deep_vitals_4H_organized.csv', 'Vital'),
        'labs': (base_path / 'deep_labs_4H_organized.csv', 'Laboratory')
    }
    
    all_results = []
    
    # Analyze each file
    for file_key, (file_path, data_type) in files_to_analyze.items():
        print(f"\n📁 Processing {file_key.upper()}")
        print("-" * 30)
        
        if file_path.exists():
            if file_key == 'static':
                results = analyze_static_variables(file_path, valid_records)
            else:
                results = analyze_longitudinal_variables(file_path, data_type, valid_records)
            
            all_results.extend(results)
        else:
            print(f"❌ File not found: {file_path}")
    
    # Generate final results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Variable', 'Data_modality'])
        
        # Define modality order: Static, Vitals, Labs
        modality_order = ['Static', 'Longitudinal (Vital)', 'Longitudinal (Laboratory)']
        df['modality_order'] = df['Data_modality'].map({mod: i for i, mod in enumerate(modality_order)})
        
        # Sort by modality order, then by record count (descending)
        df = df.sort_values(['modality_order', 'Total_records'], ascending=[True, False])
        
        # Remove helper columns
        df = df.drop(['modality_order'], axis=1)
        
        # Save results
        output_path = Path(__file__).parent / 'processed_sepsis_data_description_exact_filter.csv'
        df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE!")
        print(f"📊 Total variables analyzed: {len(df)}")
        print(f"📝 Filters applied: {note_type} notes + exclude UN labels")
        print(f"💾 Results saved to: {output_path}")
        
        # Show summary
        print(f"\n📋 Summary by Data Modality:")
        summary = df.groupby('Data_modality')['Variable'].count().sort_values(ascending=False)
        for modality, count in summary.items():
            print(f"  • {modality}: {count} variables")
        
        # Show sample variables
        print(f"\n🔬 Sample Variables by Modality:")
        for modality in df['Data_modality'].unique():
            modality_vars = df[df['Data_modality'] == modality]['Variable'].head(5)
            print(f"\n{modality}:")
            for var in modality_vars:
                record_info = df[(df['Data_modality'] == modality) & (df['Variable'] == var)].iloc[0]
                print(f"  • {var} ({record_info['Data_type']}) - {record_info['Availability_rate']} available")
        
        return df
    else:
        print("❌ No data could be analyzed!")
        return None

if __name__ == "__main__":
    main()
