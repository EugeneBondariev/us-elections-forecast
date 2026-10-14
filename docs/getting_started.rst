Getting started
===============

Install and fetch data::

    python -m pip install -r requirements.txt
    python -m pip install -e .
    python -m src.data.fetch_data

Run the pipeline::

    python -m src.models.predict         # CLI forecast + MLflow logging
    python -m src.visualization.run_analysis  # full diagnostic pipeline -> reports/
    streamlit run app.py                 # dashboard at localhost:8501
    uvicorn api:app --reload             # REST API at localhost:8000

See the top-level ``README.md`` for full details, including each data file's licensing
situation and the optional fundraising/Cook-ratings/indicator steps.
