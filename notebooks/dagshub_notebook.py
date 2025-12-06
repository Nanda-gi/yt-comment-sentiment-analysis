import dagshub
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/Nanda-gi/yt-comment-sentiment-analysis.mlflow")
dagshub.init(repo_owner='Nanda-gi', repo_name='yt-comment-sentiment-analysis', mlflow=True)
with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)