# AWS Serverless Data Pipeline for Customer Churn Analysis

## 📌 Overview

This project implements an end-to-end serverless data pipeline in AWS to analyze and predict customer churn in the telecommunications industry.

The solution integrates data engineering, analytics, and machine learning to transform raw data into actionable insights and predictive capabilities.

---

## 🧠 Problem Statement

Customer churn has a direct impact on revenue and increases customer acquisition costs. Organizations need data-driven approaches to understand customer behavior and anticipate potential churn.

---

## 🏗️ Architecture

The pipeline is built using AWS managed services:

- Amazon S3 (Data Lake: raw & processed layers)
- AWS Glue (Data Catalog + ETL with PySpark)
- Amazon Athena (SQL-based analytics)
- Amazon SageMaker (Machine Learning model)
- VPC + S3 Endpoint (secure ML execution)

---

## 🔄 Pipeline Flow

1. Data ingestion into S3 (raw layer)
2. Schema detection using AWS Glue Crawler
3. Data transformation with AWS Glue ETL
4. Storage in Parquet format (processed layer)
5. Analytical queries with Athena
6. Model training and evaluation in SageMaker

---

## 📊 Key Insights

- Customers with **monthly contracts** show higher churn rates  
- **Early-stage customers** are more likely to churn  
- Higher monthly charges correlate with higher churn  

---

## 🤖 Machine Learning Model

- Algorithm: XGBoost
- Platform: Amazon SageMaker
- AUC: **0.8440**

### Model Performance

- Strong discrimination capability (AUC > 0.84)
- Moderate recall (~55%) for churn detection
- Suitable as a proof of concept for predictive analytics

---

## 💼 Business Value

- Identification of high-risk customers  
- Improved customer segmentation  
- Support for retention strategies  
- Foundation for predictive decision-making  

---

## 🔐 Security

The ML component was deployed within a VPC using private subnets and an S3 gateway endpoint, ensuring secure access to data without relying on public internet connectivity.

---

## 📁 Repository Structure

aws-churn-data-pipeline/

├── README.md

├── architecture/

│   └── aws-architecture-diagram.png

├── scripts/

│   ├── glue_data_exploration_sagemaker_churn_model.py

│   └── glue_etl_job.py

├── results/

│   ├── roc_curve.png

│   └── confusion_matrix.png

└── docs/

    └── final_report.pdf


---

## 🚀 Conclusion

This project demonstrates how modern data architectures can evolve from descriptive analytics to predictive capabilities, enabling organizations to make better data-driven decisions.

---

## 👤 Author

César Augusto Amórtegui Hernández
