# Hotel Booking Cancellation Business Analytics

A business analytics and predictive modelling project focused on understanding hotel booking cancellation behaviour and identifying operational strategies to reduce cancellation risk.

---

## Project Overview

Hotel booking cancellations create significant operational and revenue uncertainty for hotels. This project analyses booking behaviour across City Hotel and Resort Hotel segments to identify key drivers of cancellation risk and develop predictive models for cancellation monitoring.

The analysis combines:

- Exploratory Business Analysis
- Customer and operational behaviour analysis
- Predictive modelling
- Business interpretation
- Management recommendations

This project is structured as a practical business analytics report rather than a purely academic machine learning notebook.

---

## Business Problem

Hotel operators face several challenges caused by booking cancellations:

- Revenue instability
- Inefficient room allocation
- Staffing uncertainty
- Demand forecasting issues
- Customer communication inefficiencies

This project investigates questions such as:

- Which hotel type experiences higher cancellation risk?
- Which customer groups are more likely to cancel?
- How do booking channels and deposit policies affect cancellations?
- Can booking-level information predict cancellation likelihood before arrival?

---

## Dataset

Dataset: **Hotel Booking Demand Dataset**

The dataset contains reservation-level records for both City Hotel and Resort Hotel bookings, including:

- Lead time
- Customer type
- Booking channel
- Deposit policy
- Room type
- Market segment
- Country
- ADR (Average Daily Rate)
- Repeated guest behaviour
- Booking changes
- Cancellation outcome

### Target Variable

```python
is_canceled
```

- `1` = cancelled booking
- `0` = completed booking

---

## Business Analytics Workflow

The project follows a business analytics workflow:

1. Data Understanding
2. Exploratory Business Analysis
3. Data Preparation
4. Predictive Modelling
5. Business Interpretation
6. Management Recommendations

---

## Key Business Insights

### City Hotel shows higher cancellation risk

City Hotel records substantially higher cancellation rates than Resort Hotel, indicating greater operational uncertainty.

### Seasonal demand patterns are significant

Booking demand peaks during July and August, suggesting opportunities for seasonal pricing and staffing optimisation.

### New guests cancel more frequently

Repeated guests show much lower cancellation rates, highlighting the value of customer retention strategies.

### Deposit policy strongly affects cancellations

Non-refundable bookings exhibit extremely high cancellation rates, suggesting possible behavioural or policy-related issues.

### Room type differences matter

Certain room categories experience higher cancellation rates and may require improved pricing and inventory management.

---

## Predictive Modelling

Three classification models were developed and compared:

| Model | Accuracy | AUC Score |
|---|---|---|
| Random Forest | 88.9% | 0.95 |
| XGBoost | 87.4% | 0.95 |
| Logistic Regression | 79.6% | 0.87 |

### Model Interpretation

- Random Forest achieved the strongest overall predictive performance.
- XGBoost produced similarly strong classification capability.
- Logistic Regression provided a more interpretable but weaker baseline model.

The results suggest that hotel cancellation behaviour contains complex non-linear relationships that are better captured by tree-based models.

---

## Management Recommendations

The analysis proposes several practical business actions:

- Strengthen cancellation prevention strategies for City Hotel
- Improve first-time guest communication
- Optimise deposit and payment policy structures
- Improve room information transparency
- Apply seasonal pricing and staffing strategies
- Develop predictive cancellation risk monitoring workflows

---

## Technologies Used

### Programming & Analytics

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

### Methods

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Classification Modelling
- ROC/AUC Evaluation
- Cross Validation

---

## Repository Structure

```text
hotel-booking-cancellation-analysis/
│
├── hotel_booking_business_analytics_report.ipynb
├── hotel_booking_business_analytics_report.html
├── README.md
├── images/
└── data/
```

---

## Project Outputs

This repository includes:

- Business analytics report notebook
- Visual business analysis
- Predictive modelling results
- ROC curve evaluation
- Management recommendations
- Operational business interpretation

---

## Future Improvements

Potential future enhancements include:

- Real-time cancellation risk scoring
- Dashboard integration
- Explainable AI techniques (SHAP / feature importance)
- Hyperparameter optimisation
- Deployment as a hotel decision-support application

---

## Author

**Yang Li**

---
