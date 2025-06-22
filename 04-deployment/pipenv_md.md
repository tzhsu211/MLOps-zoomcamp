| Task                            | Command                                                         |
| ------------------------------- | --------------------------------------------------------------- |
| Initialize env                  | `pipenv --python 3.12`                                          |
| Install packages                | `pipenv install flask mlflow xgboost scikit-learn pandas numpy` |
| Add dev package (e.g. requests) | `pipenv install --dev requests`                                 |
| Run app/test                    | `pipenv shell` → `python predict.py` / `python test.py`         |
