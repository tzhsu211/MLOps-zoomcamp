From Micorsoft Copilot

# MLflow Quick Guide

## 1. How to Start MLflow
MLflow requires setting up a **tracking URI** in both your **code** and **CLI commands** to ensure experiments are logged correctly.

### ✅ Step-by-Step Setup
1. **Install MLflow (if not already installed)**
    ```sh
    pip install mlflow
    ```

2. **Set Tracking URI in Code**
    ```python
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Store logs locally
    mlflow.set_experiment("MyExperiment")
    ```

3. **Start MLflow Server (For Remote Tracking)**
    ```sh
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    ```
    - Enables **remote tracking** and allows multiple users to access MLflow.

4. **Start MLflow UI (Without Server)**
    ```sh
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
    - Launches **only the UI** without enabling remote tracking.

5. **Access MLflow Dashboard**
    - Open browser: **`http://localhost:5000`**
    - View experiment logs, metrics, and artifacts.

---

## 2. Difference Between MLflow Server & UI-Only Mode
| Feature | MLflow Server (`mlflow server`) | UI Only (`mlflow ui`) |
|---------|---------------------------------|----------------------|
| **Purpose** | Tracks experiments **remotely** | View **local** logs only |
| **Remote Access** | ✅ Yes (multiple users can log runs) | ❌ No (logs stay local) |
| **Database Usage** | Stores in **SQLite, PostgreSQL, or S3** | Only reads local experiment files |
| **Runs on a Web Server?** | ✅ Yes | ❌ No (purely a UI) |

🔹 If working **solo**, `mlflow ui` is usually enough.  
🔹 If collaborating, use `mlflow server` for shared experiment tracking.

---

## 3. Common MLflow Commands
| **Command** | **Usage** |
|------------|-----------|
| `mlflow.sklearn.autolog()` | Automatically logs model parameters, metrics, and artifacts. |
| `mlflow.start_run()` | Starts a new MLflow run for tracking. |
| `mlflow.log_param("name", value)` | Logs a parameter (e.g., `mlflow.log_param("learning_rate", 0.01)`). |
| `mlflow.log_metric("name", value)` | Logs a metric (e.g., `mlflow.log_metric("rmse", 2.3)`). |
| `mlflow.log_artifact("path/to/file")` | Saves a file (e.g., dataset, model config) into MLflow. |
| `mlflow.sklearn.log_model(model, "model_name")` | Stores a trained Scikit-Learn model in MLflow. |
| `mlflow.set_experiment("experiment_name")` | Specifies the experiment name. |
| `mlflow.list_experiments()` | Lists all experiments in MLflow. |

---

## Summary
✅ **MLflow tracks experiments**, logs metrics, and saves models.  
✅ **Set `mlflow.set_tracking_uri()` in code and CLI** to ensure logs are stored properly.  
✅ **Use `mlflow server` for team collaboration, or `mlflow ui` for solo work.**  
✅ **Leverage MLflow commands (`log_param`, `log_metric`, `autolog`) for automation.**  

Let me know if you'd like extra details on **deploying models** or **integrating MLflow with Prefect!** 🚀
