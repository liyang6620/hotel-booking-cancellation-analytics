"""Shared paths and point-in-time feature contract."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "2022_hotel_bookings.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

MODEL_PATH = ARTIFACTS_DIR / "cancellation_model.ubj"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
CALIBRATOR_PATH = ARTIFACTS_DIR / "probability_calibrator.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
FEATURE_CONTRACT_PATH = ARTIFACTS_DIR / "feature_contract.json"

TARGET = "is_canceled"
TRAIN_END = "2016-09-01"
VALIDATION_END = "2017-01-01"
RANDOM_STATE = 42

DIRECT_LEAKAGE_COLUMNS = ("reservation_status", "reservation_status_date")
POST_BOOKING_COLUMNS = ("assigned_room_type", "booking_changes", "days_in_waiting_list")

CATEGORICAL_FEATURES = (
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent_code",
)

NUMERIC_FEATURES = (
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_nights",
    "total_guests",
    "is_family",
    "has_agent",
    "has_company",
    "arrival_month_sin",
    "arrival_month_cos",
    "arrival_week_sin",
    "arrival_week_cos",
)

RAW_REQUIRED_COLUMNS = (
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
)
