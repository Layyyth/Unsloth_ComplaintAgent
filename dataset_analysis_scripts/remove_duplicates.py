import pandas as pd
import numpy as np
import json
import os
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class DuplicateRemover:
    def __init__(self, data_path):
        """Initialize with data loading"""
        self.df = self.load_data(data_path)
        self.original_count = len(self.df)
        print(f"Loaded dataset with {self.original_count} rows")
    
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
                ticket_data = sample.get('ticket_data', {})
                if isinstance(ticket_data, dict):
                    for k, v in ticket_data.items():
                        row[k] = v
                records.append(row)
            return pd.DataFrame(records)
        elif data_path.endswith('.jsonl'):
            return pd.read_json(data_path, lines=True)
        else:
            raise ValueError("Unsupported file format")
    
    def exact_duplicate_removal(self):
        """Remove exact duplicates based on all columns"""
        print("\n" + "=" * 60)
        print("EXACT DUPLICATE REMOVAL")
        print("=" * 60)
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        final_count = len(self.df)
        removed_count = initial_count - final_count
        
        print(f"Initial rows: {initial_count}")
        print(f"After exact deduplication: {final_count}")
        print(f"Removed: {removed_count} rows ({removed_count/initial_count*100:.2f}%)")
        
        return removed_count
    
    def template_based_deduplication(self, text_columns=['user_input', 'title', 'description']):
        """Remove duplicates based on text similarity using templates"""
        print("\n" + "=" * 60)
        print("TEMPLATE-BASED DEDUPLICATION")
        print("=" * 60)
        
        # Find existing text columns
        existing_cols = [col for col in text_columns if col in self.df.columns]
        
        if not existing_cols:
            print("No text columns found for template-based deduplication")
            return 0
        
        print(f"Using columns: {existing_cols}")
        
        # Combine text columns for similarity analysis
        self.df['combined_text'] = self.df[existing_cols].astype(str).agg(' '.join, axis=1)
        
        # Clean and normalize text
        self.df['cleaned_text'] = self.df['combined_text'].apply(self.normalize_text)
        
        # Create templates by grouping similar texts
        templates = self.create_text_templates()
        
        # Remove duplicates based on templates
        initial_count = len(self.df)
        self.df = self.remove_template_duplicates(templates)
        final_count = len(self.df)
        removed_count = initial_count - final_count
        
        print(f"Initial rows: {initial_count}")
        print(f"After template deduplication: {final_count}")
        print(f"Removed: {removed_count} rows ({removed_count/initial_count*100:.2f}%)")
        print(f"Created {len(templates)} templates")
        
        # Clean up temporary columns
        self.df = self.df.drop(['combined_text', 'cleaned_text'], axis=1, errors='ignore')
        
        return removed_count
    
    def normalize_text(self, text):
        """Normalize text for better comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (keep spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words).strip()
    
    def create_text_templates(self, similarity_threshold=0.8):
        """Create templates by grouping similar texts"""
        print(f"Creating templates with similarity threshold: {similarity_threshold}")
        
        # Use TF-IDF for text similarity
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform the cleaned texts
        tfidf_matrix = vectorizer.fit_transform(self.df['cleaned_text'])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Group similar texts into templates
        templates = []
        used_indices = set()
        
        for i in range(len(self.df)):
            if i in used_indices:
                continue
            
            # Find similar texts
            similar_indices = []
            for j in range(i, len(self.df)):
                if similarity_matrix[i][j] >= similarity_threshold:
                    similar_indices.append(j)
                    used_indices.add(j)
            
            if len(similar_indices) > 1:  # Only create templates for groups
                templates.append({
                    'representative_idx': i,
                    'similar_indices': similar_indices,
                    'count': len(similar_indices),
                    'representative_text': self.df.iloc[i]['cleaned_text']
                })
        
        # Sort templates by frequency
        templates.sort(key=lambda x: x['count'], reverse=True)
        
        return templates
    
    def remove_template_duplicates(self, templates):
        """Remove duplicates based on templates, keeping the best representative"""
        print("Removing template duplicates...")
        
        # Keep track of rows to remove
        rows_to_remove = set()
        
        for template in templates:
            # Keep the first occurrence, remove the rest
            similar_indices = template['similar_indices'][1:]  # Skip the representative
            rows_to_remove.update(similar_indices)
        
        # Remove duplicate rows - use iloc to avoid index issues
        valid_indices = [i for i in range(len(self.df)) if i not in rows_to_remove]
        df_cleaned = self.df.iloc[valid_indices].copy()
        
        return df_cleaned
    
    def semantic_deduplication(self, text_column='user_input', similarity_threshold=0.85):
        """Remove semantic duplicates using advanced text similarity"""
        print("\n" + "=" * 60)
        print("SEMANTIC DEDUPLICATION")
        print("=" * 60)
        
        if text_column not in self.df.columns:
            print(f"Column '{text_column}' not found")
            return 0
        
        print(f"Using column: {text_column}")
        print(f"Similarity threshold: {similarity_threshold}")
        
        # Clean and normalize text
        self.df['semantic_text'] = self.df[text_column].apply(self.normalize_text)
        
        # Use TF-IDF for semantic similarity
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        
        tfidf_matrix = vectorizer.fit_transform(self.df['semantic_text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Remove semantic duplicates
        initial_count = len(self.df)
        self.df = self.remove_semantic_duplicates(similarity_matrix, similarity_threshold)
        final_count = len(self.df)
        removed_count = initial_count - final_count
        
        print(f"Initial rows: {initial_count}")
        print(f"After semantic deduplication: {final_count}")
        print(f"Removed: {removed_count} rows ({removed_count/initial_count*100:.2f}%)")
        
        # Clean up temporary column
        self.df = self.df.drop(['semantic_text'], axis=1, errors='ignore')
        
        return removed_count
    
    def remove_semantic_duplicates(self, similarity_matrix, threshold):
        """Remove semantic duplicates keeping the best representative"""
        rows_to_keep = []
        rows_to_remove = set()
        
        for i in range(len(self.df)):
            if i in rows_to_remove:
                continue
            
            rows_to_keep.append(i)
            
            # Find similar rows to remove
            for j in range(i + 1, len(self.df)):
                if j not in rows_to_remove and similarity_matrix[i][j] >= threshold:
                    rows_to_remove.add(j)
        
        # Use iloc to avoid index issues
        return self.df.iloc[rows_to_keep].copy()
    
    def column_based_deduplication(self, key_columns):
        """Remove duplicates based on specific key columns"""
        print("\n" + "=" * 60)
        print("COLUMN-BASED DEDUPLICATION")
        print("=" * 60)
        
        # Find existing key columns
        existing_cols = [col for col in key_columns if col in self.df.columns]
        
        if not existing_cols:
            print("No key columns found")
            return 0
        
        print(f"Using key columns: {existing_cols}")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=existing_cols)
        final_count = len(self.df)
        removed_count = initial_count - final_count
        
        print(f"Initial rows: {initial_count}")
        print(f"After column-based deduplication: {final_count}")
        print(f"Removed: {removed_count} rows ({removed_count/initial_count*100:.2f}%)")
        
        return removed_count
    
    def comprehensive_deduplication(self, output_path=None):
        """Run comprehensive deduplication pipeline"""
        print("🧹 Starting Comprehensive Deduplication Pipeline")
        print("=" * 80)
        
        total_removed = 0
        
        # Step 1: Remove exact duplicates
        removed = self.exact_duplicate_removal()
        total_removed += removed
        
        # Step 2: Remove duplicates based on key columns
        key_columns = ['user_input', 'ticket_type', 'severity']
        removed = self.column_based_deduplication(key_columns)
        total_removed += removed
        
        # Step 3: Template-based deduplication
        removed = self.template_based_deduplication()
        total_removed += removed
        
        # Step 4: Semantic deduplication
        removed = self.semantic_deduplication()
        total_removed += removed
        
        # Final statistics
        final_count = len(self.df)
        total_reduction = self.original_count - final_count
        
        print("\n" + "=" * 80)
        print("DEDUPLICATION SUMMARY")
        print("=" * 80)
        print(f"Original dataset: {self.original_count} rows")
        print(f"Final dataset: {final_count} rows")
        print(f"Total removed: {total_reduction} rows ({total_reduction/self.original_count*100:.2f}%)")
        print(f"Data reduction: {total_reduction/self.original_count*100:.1f}%")
        
        # Save cleaned dataset
        if output_path:
            self.save_cleaned_data(output_path)
            print(f"✅ Cleaned dataset saved to: {output_path}")
        
        return self.df
    
    def save_cleaned_data(self, output_path):
        """Save the cleaned dataset"""
        if output_path.endswith('.json'):
            # Convert back to the original nested format
            records = []
            for _, row in self.df.iterrows():
                ticket_data = {
                    'ticket_type': row['ticket_type'],
                    'title': row['title'],
                    'description': row['description'],
                    'severity': row['severity'],
                    'department_impacted': row['department_impacted'],
                    'service_impacted': row['service_impacted'],
                    'preferred_communication': row['preferred_communication']
                }
                
                # Add assistance_request only if it exists
                if 'assistance_request' in row:
                    ticket_data['assistance_request'] = row['assistance_request']
                
                record = {
                    'user_input': row['user_input'],
                    'ticket_data': ticket_data
                }
                records.append(record)
            
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)
        
        elif output_path.endswith('.csv'):
            self.df.to_csv(output_path, index=False)
        
        else:
            raise ValueError("Unsupported output format. Use .json or .csv")

# Usage Example
if __name__ == "__main__":
    # Path to the training data JSON file
    input_path = "datasets/new_training_data.json"
    output_path = "datasets/new_training_data_cleaned.json"
    
    # Initialize deduplicator
    deduplicator = DuplicateRemover(input_path)
    
    # Run comprehensive deduplication
    cleaned_df = deduplicator.comprehensive_deduplication(output_path)
    
    print("\n" + "=" * 80)
    print("✅ Deduplication Complete!")
    print("=" * 80) 