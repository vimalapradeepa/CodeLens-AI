$body = '{"full_name": "apache/airflow"}'
Invoke-WebRequest -Uri "http://127.0.0.1:5000/api/analyze" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing