import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter
import re
import json
from textstat import flesch_reading_ease, flesch_kincaid_grade
import warnings
import os
import io
import sys
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComplaintDataEDA:
    def __init__(self, data_path):
        """Initialize EDA class with data loading"""
        self.df = self.load_data(data_path)
        self.setup_plots()
    
    def load_data(self, data_path):
        """Load data from various formats"""
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            # Handle nested JSON structure for training data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Flatten the nested structure
            records = []
            for sample in data:
                row = {'user_input': sample.get('user_input', '')}
                
                # Handle enhanced dataset with both ticket_data and root level fields
                ticket_data = sample.get('ticket_data', {})
                if isinstance(ticket_data, dict):
                    for k, v in ticket_data.items():
                        row[k] = v
                
                # Also include root level fields (for enhanced dataset)
                for k, v in sample.items():
                    if k not in ['user_input', 'ticket_data']:
                        row[k] = v
                
                records.append(row)
            return pd.DataFrame(records)
        elif data_path.endswith('.jsonl'):
            return pd.read_json(data_path, lines=True)
        else:
            raise ValueError("Unsupported file format")
    
    def setup_plots(self):
        """Setup plotting configurations"""
        # Set matplotlib to use non-interactive backend to avoid display issues
        import matplotlib
        matplotlib.use('Agg')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def basic_info(self):
        """Display basic dataset information"""
        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print("\nColumn Information:")
        print(self.df.info())
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])
        print("\nDuplicate Rows:", self.df.duplicated().sum())
    
    def text_analysis(self, text_columns=['subject', 'body', 'description']):
        """Analyze text features"""
        print("\n" + "=" * 60)
        print("TEXT ANALYSIS")
        print("=" * 60)
        
        # Find existing text columns
        existing_cols = [col for col in text_columns if col in self.df.columns]
        
        if not existing_cols:
            print("No text columns found in the dataset")
            return
        
        # Handle different subplot configurations based on number of columns
        if len(existing_cols) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            axes = [[ax]]
        elif len(existing_cols) <= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes = [axes]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        fig.suptitle('Text Feature Analysis', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(existing_cols[:4]):
            if col in self.df.columns:
                # Text length distribution
                text_lengths = self.df[col].astype(str).str.len()
                
                if len(existing_cols) == 1:
                    ax = axes[0][0]
                elif len(existing_cols) <= 2:
                    ax = axes[0][i]
                else:
                    ax = axes[i//2, i%2]
                
                ax.hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'{col.title()} Length Distribution')
                ax.set_xlabel('Character Count')
                ax.set_ylabel('Frequency')
                
                # Add statistics
                ax.axvline(text_lengths.mean(), color='red', linestyle='--', 
                          label=f'Mean: {text_lengths.mean():.1f}')
                ax.axvline(text_lengths.median(), color='green', linestyle='--', 
                          label=f'Median: {text_lengths.median():.1f}')
                ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Text statistics
        for col in existing_cols:
            if col in self.df.columns:
                texts = self.df[col].astype(str)
                word_counts = texts.str.split().str.len()
                
                print(f"\n{col.upper()} Statistics:")
                print(f"  Average length: {texts.str.len().mean():.1f} characters")
                print(f"  Average words: {word_counts.mean():.1f}")
                print(f"  Max length: {texts.str.len().max()} characters")
                print(f"  Min length: {texts.str.len().min()} characters")
    
    def categorical_analysis(self):
        """Analyze categorical variables"""
        print("\n" + "=" * 60)
        print("CATEGORICAL ANALYSIS")
        print("=" * 60)
        
        # Common categorical columns in complaint datasets
        cat_columns = ['priority', 'queue', 'type', 'department', 'service', 
                      'category', 'subcategory', 'status', 'tags']
        
        existing_cats = [col for col in cat_columns if col in self.df.columns]
        
        if not existing_cats:
            # Try to identify categorical columns automatically
            existing_cats = self.df.select_dtypes(include=['object']).columns.tolist()
            existing_cats = [col for col in existing_cats if self.df[col].nunique() < 50]
        
        if existing_cats:
            n_cols = min(3, len(existing_cats))
            n_rows = (len(existing_cats) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            # Handle single subplot case
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(existing_cats):
                value_counts = self.df[col].value_counts()
                
                ax = axes[i] if len(existing_cats) > 1 else axes[0]
                value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'{col.title()} Distribution')
                ax.set_xlabel(col.title())
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                total = len(self.df)
                for j, v in enumerate(value_counts.values):
                    ax.text(j, v + total*0.01, f'{v/total*100:.1f}%', 
                           ha='center', va='bottom', fontsize=8)
            
            # Hide empty subplots
            for i in range(len(existing_cats), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
            
            # Print detailed statistics
            for col in existing_cats:
                print(f"\n{col.upper()} Distribution:")
                value_counts = self.df[col].value_counts()
                percentages = (value_counts / len(self.df) * 100).round(2)
                
                for value, count, pct in zip(value_counts.index, value_counts.values, percentages):
                    print(f"  {value}: {count} ({pct}%)")
                
                print(f"  Unique values: {self.df[col].nunique()}")
                print(f"  Most common: {value_counts.index[0]} ({percentages.iloc[0]}%)")
    
    def check_data_leakage(self):
        """Check for the specific data leakage issue mentioned"""
        print("\n" + "=" * 60)
        print("DATA LEAKAGE ANALYSIS")
        print("=" * 60)
        
        # Check for uniform values in service/department columns
        leakage_columns = ['service', 'department', 'impacted_service', 'impacted_department']
        existing_leakage_cols = [col for col in leakage_columns if col in self.df.columns]
        
        for col in existing_leakage_cols:
            unique_values = self.df[col].nunique()
            total_rows = len(self.df)
            
            print(f"\n{col.upper()}:")
            print(f"  Unique values: {unique_values}")
            print(f"  Total rows: {total_rows}")
            
            if unique_values == 1:
                print(f"  ⚠️  WARNING: All values are identical!")
                print(f"  Value: {self.df[col].iloc[0]}")
            elif unique_values < 5:
                print(f"  ⚠️  WARNING: Very low diversity!")
                print("  Distribution:")
                for value, count in self.df[col].value_counts().items():
                    print(f"    {value}: {count} ({count/total_rows*100:.1f}%)")
    
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def temporal_analysis(self):
        """Analyze temporal patterns if date columns exist"""
        print("\n" + "=" * 60)
        print("TEMPORAL ANALYSIS")
        print("=" * 60)
        
        # Look for date columns
        date_columns = []
        for col in self.df.columns:
            if any(word in col.lower() for word in ['date', 'time', 'created', 'updated']):
                try:
                    pd.to_datetime(self.df[col])
                    date_columns.append(col)
                except:
                    continue
        
        if not date_columns:
            print("No date columns found")
            return
        
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col])
            
            # Plot time series
            plt.figure(figsize=(12, 6))
            
            # Group by date and count
            daily_counts = self.df.groupby(self.df[col].dt.date).size()
            daily_counts.plot(kind='line', marker='o')
            plt.title(f'Complaints Over Time ({col})')
            plt.xlabel('Date')
            plt.ylabel('Number of Complaints')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def generate_wordclouds(self, text_columns=['subject', 'body', 'description']):
        """Generate word clouds for text columns"""
        print("\n" + "=" * 60)
        print("WORD CLOUD ANALYSIS")
        print("=" * 60)
        
        existing_cols = [col for col in text_columns if col in self.df.columns]
        
        for col in existing_cols:
            if col in self.df.columns:
                # Combine all text
                text = ' '.join(self.df[col].astype(str).values)
                
                # Clean text
                text = re.sub(r'[^\w\s]', ' ', text.lower())
                text = re.sub(r'\s+', ' ', text)
                
                # Generate word cloud
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    max_words=100,
                                    colormap='viridis').generate(text)
                
                plt.figure(figsize=(12, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud - {col.title()}', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.show()
    
    def data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        
        quality_issues = []
        
        # Check for missing values
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 10]
        if len(high_missing) > 0:
            quality_issues.append(f"High missing values: {dict(high_missing)}")
        
        # Check for duplicates
        duplicate_pct = (self.df.duplicated().sum() / len(self.df) * 100).round(2)
        if duplicate_pct > 5:
            quality_issues.append(f"High duplicate rate: {duplicate_pct}%")
        
        # Check for low cardinality
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].nunique() == 1:
                quality_issues.append(f"No variation in {col}")
        
        # Summary
        if quality_issues:
            print("⚠️  QUALITY ISSUES FOUND:")
            for issue in quality_issues:
                print(f"  - {issue}")
        else:
            print("✅ No major quality issues detected")
        
        # Data completeness
        completeness = (1 - self.df.isnull().sum() / len(self.df)) * 100
        print(f"\nData Completeness by Column:")
        for col, comp in completeness.items():
            status = "✅" if comp > 95 else "⚠️" if comp > 80 else "❌"
            print(f"  {col}: {comp:.1f}% {status}")
    
    def analyze_duplicates(self, output_dir="eda_results"):
        """Analyze and save all duplicates to a text file"""
        print("\n" + "=" * 60)
        print("DUPLICATE ANALYSIS")
        print("=" * 60)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all duplicates
        duplicates = self.df[self.df.duplicated(keep=False)]
        duplicate_count = len(duplicates)
        total_count = len(self.df)
        duplicate_pct = (duplicate_count / total_count) * 100
        
        print(f"Total rows: {total_count}")
        print(f"Duplicate rows: {duplicate_count} ({duplicate_pct:.2f}%)")
        print(f"Unique rows: {total_count - duplicate_count}")
        
        if duplicate_count > 0:
            # Group duplicates by their content
            duplicate_groups = duplicates.groupby(duplicates.columns.tolist()).size().reset_index(name='count')
            duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1].sort_values('count', ascending=False)
            
            print(f"\nFound {len(duplicate_groups)} unique duplicate patterns")
            
            # Save detailed duplicate analysis to file
            output_file = os.path.join(output_dir, "duplicate_analysis.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("DUPLICATE ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Dataset: {total_count} total rows\n")
                f.write(f"Duplicates: {duplicate_count} rows ({duplicate_pct:.2f}%)\n")
                f.write(f"Unique rows: {total_count - duplicate_count}\n")
                f.write(f"Duplicate patterns: {len(duplicate_groups)}\n\n")
                
                f.write("DUPLICATE PATTERNS (sorted by frequency):\n")
                f.write("-" * 50 + "\n\n")
                
                for idx, row in duplicate_groups.head(20).iterrows():
                    f.write(f"Pattern #{idx + 1} (appears {row['count']} times):\n")
                    for col in self.df.columns:
                        if col != 'count':
                            f.write(f"  {col}: {row[col]}\n")
                    f.write("\n")
                
                if len(duplicate_groups) > 20:
                    f.write(f"... and {len(duplicate_groups) - 20} more patterns\n\n")
                
                # Show some examples of the most common duplicates
                f.write("EXAMPLES OF MOST COMMON DUPLICATES:\n")
                f.write("-" * 50 + "\n\n")
                
                for idx, row in duplicate_groups.head(5).iterrows():
                    f.write(f"Example {idx + 1} (Frequency: {row['count']} times):\n")
                    f.write("-" * 30 + "\n")
                    for col in self.df.columns:
                        if col != 'count':
                            f.write(f"{col}: {row[col]}\n")
                    f.write("\n")
            
            print(f"✅ Detailed duplicate analysis saved to: {output_file}")
            
            # Show summary of most common duplicates
            print(f"\nTop 5 most common duplicate patterns:")
            for idx, row in duplicate_groups.head(5).iterrows():
                print(f"  Pattern #{idx + 1}: {row['count']} occurrences")
                # Show first few characters of user_input for identification
                user_input = str(row['user_input'])[:100] + "..." if len(str(row['user_input'])) > 100 else str(row['user_input'])
                print(f"    User input: {user_input}")
                print(f"    Ticket type: {row['ticket_type']}")
                print(f"    Severity: {row['severity']}")
                print()
        else:
            print("✅ No duplicates found in the dataset")
    
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("🔍 Starting Comprehensive EDA for Complaint Dataset")
        print("=" * 80)
        
        self.basic_info()
        self.text_analysis()
        self.categorical_analysis()
        self.check_data_leakage()
        self.correlation_analysis()
        self.temporal_analysis()
        self.generate_wordclouds()
        self.data_quality_report()
        self.analyze_duplicates()
        
        print("\n" + "=" * 80)
        print("✅ EDA Complete! Review the outputs above for insights.")
        print("=" * 80)

# Usage Example
if __name__ == "__main__":
    # Path to the enhanced training data JSON file
    data_path = "datasets/new_training_data_cleaned.json"

    # --- Begin capturing output ---
    output_buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = output_buffer

    # Initialize and run EDA
    eda = ComplaintDataEDA(data_path)
    eda.run_full_eda()

    # --- End capturing output ---
    sys.stdout = sys_stdout
    eda_output = output_buffer.getvalue()

    # Ensure eda_results directory exists
    results_dir = "eda_results"
    os.makedirs(results_dir, exist_ok=True)

    # Find next available file number
    existing_files = [f for f in os.listdir(results_dir) if f.startswith("eda_output_") and f.endswith(".txt")]
    numbers = [int(f.split("_")[2].split(".")[0]) for f in existing_files if f.split("_")[2].split(".")[0].isdigit()]
    next_num = max(numbers) + 1 if numbers else 1
    output_path = os.path.join(results_dir, f"eda_output_{next_num}.txt")

    # Write output to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(eda_output)

    print(f"\nEDA results saved to {output_path}\n")
    
    # Or run individual analyses
    # eda.basic_info()
    # eda.check_data_leakage()
    # eda.categorical_analysis()
