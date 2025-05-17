#!/usr/bin/env python3
"""
convert.py - Script to load data from SQLite and import into Qdrant vector database
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configuration parameters
SQLITE_DB_PATH = "data/california_schools/california_schools.sqlite"  # Replace with your SQLite DB path
TABLE_NAME = "schools"  # Replace with your table name
QDRANT_HOST = "localhost"  # Default Qdrant host
QDRANT_PORT = 6333  # Default Qdrant port
COLLECTION_NAME = "school_vectors"  # Name for your Qdrant collection
EMBEDDING_DIM = 384  # Dimension for embedding model (depends on model you choose)
BATCH_SIZE = 100  # Process records in batches to manage memory

def get_db_connection(db_path):
    """Create a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        print(f"Successfully connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        if conn:
            conn.close()
        raise

def get_embedding_model():
    """Load the sentence transformer embedding model."""
    # Using a general-purpose model, but you can change to one more specific to your needs
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_text_for_embedding(row):
    """Create a text representation of a data row for embedding."""
    # Combine relevant fields - adjust based on what's important for your search needs
    text_parts = []
    
    # Add school/district identification info
    if row['School']:
        text_parts.append(f"School: {row['School']}")
    if row['District']:
        text_parts.append(f"District: {row['District']}")
    if row['County']:
        text_parts.append(f"County: {row['County']}")
    
    # Add location info
    location_parts = []
    if row['City']:
        location_parts.append(row['City'])
    if row['State']:
        location_parts.append(row['State'])
    if location_parts:
        text_parts.append(f"Location: {', '.join(location_parts)}")
    
    # Add contact info
    if row['Website']:
        text_parts.append(f"Website: {row['Website']}")
    if row['Phone']:
        text_parts.append(f"Phone: {row['Phone']}")
    
    # Add educational information
    if row['SOCType']:
        text_parts.append(f"School Type: {row['SOCType']}")
    if row['EdOpsName']:
        text_parts.append(f"Education Option: {row['EdOpsName']}")
    if row['GSoffered']:
        text_parts.append(f"Grades Offered: {row['GSoffered']}")
    if row['GSserved']:
        text_parts.append(f"Grades Served: {row['GSserved']}")
    
    # Additional details
    if row.get('Charter') == 1:
        text_parts.append("Charter School")
    if row.get('Magnet') == 1:
        text_parts.append("Magnet School")
    
    # Add virtual instruction status
    virtual_types = {
        'F': 'Exclusively Virtual',
        'V': 'Primarily Virtual',
        'C': 'Primarily Classroom',
        'N': 'Not Virtual',
        'P': 'Partial Virtual'
    }
    if row.get('Virtual') in virtual_types:
        text_parts.append(f"Virtual Status: {virtual_types[row['Virtual']]}")
    
    return " | ".join(text_parts)

def setup_qdrant_collection(client, collection_name, embedding_dim):
    """Create or get a Qdrant collection for storing vectors."""
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if not collection_exists:
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE
            )
        )
        print(f"Created new collection: {collection_name}")
    else:
        print(f"Collection {collection_name} already exists")

def main():
    # Connect to SQLite database
    conn = get_db_connection(SQLITE_DB_PATH)
    
    # Get the embedding model
    model = get_embedding_model()
    print(f"Loaded embedding model with dimension: {model.get_sentence_embedding_dimension()}")
    
    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Set up collection
    setup_qdrant_collection(client, COLLECTION_NAME, EMBEDDING_DIM)
    
    # Query data and process in batches
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows to process: {total_rows}")
    
    # Process in batches
    offset = 0
    while offset < total_rows:
        print(f"Processing batch starting at offset {offset}")
        
        # Get a batch of data
        cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT {BATCH_SIZE} OFFSET {offset}")
        batch_data = [dict(row) for row in cursor.fetchall()]
        
        if not batch_data:
            break
        
        # Prepare data for Qdrant
        ids = []
        vectors = []
        payloads = []
        
        for i, row in enumerate(batch_data):
            # Create text representation for embedding
            text = create_text_for_embedding(row)
            
            # Generate embedding
            embedding = model.encode(text)
            
            # Add to batch data
            ids.append(offset + i)  # Unique ID for each point
            vectors.append(embedding.tolist())  # Convert numpy array to list
            
            # Create payload
            payload = {
                # Include all fields that you want to retrieve later
                "cds_code": row.get('CDSCode'),
                "school": row.get('School'),
                "district": row.get('District'),
                "county": row.get('County'),
                "city": row.get('City'),
                "state": row.get('State'),
                "zipcode": row.get('Zip'),
                "website": row.get('Website'),
                "phone": row.get('Phone'),
                "type": row.get('SOCType'),
                "education_option": row.get('EdOpsName'),
                "grades_offered": row.get('GSoffered'),
                "grades_served": row.get('GSserved'),
                "charter": bool(row.get('Charter')),
                "magnet": bool(row.get('Magnet')),
                "virtual": row.get('Virtual'),
                "latitude": row.get('Latitude'),
                "longitude": row.get('Longitude'),
                "text_representation": text  # Store the text used for embedding
            }
            payloads.append(payload)
        
        # Upload batch to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )
        )
        print(f"Uploaded {len(ids)} points to Qdrant")
        
        # Move to next batch
        offset += BATCH_SIZE
    
    print("Conversion complete!")
    conn.close()

if __name__ == "__main__":
    main()