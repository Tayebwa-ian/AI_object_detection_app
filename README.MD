# Group 5

| Name | Email | FIM ID |
| ---- | ----- | ------ |
|  Ian Tayebwa    |  tayebw01@ads.uni-passau.de  |    tayebwa    |
|  Whenton Seth   |  whento01@ads.uni-passau.de  |    Whenton    |
|  Jack Grant Dagogo  |  dagogo01@ads.uni-passau.de  |    dagogojack    |

## Run API with Development Server (runs with MYSQL server based Databse):  
- [ ] Install python and create a virtual enviroment
- [ ] Install all depedences using requirements.txt file (pip install -r requirements.txt)
- [ ] Activate the development environment
- [ ] Create a development database using the sql script at the root of this project directory 
- [ ] Run the flask development server 


## Run API with Development Server (runs with MYSQL server based Databse):  

```
OBJ_DETECT_ENV=dev OBJ_DETECT_MYSQL_USER=obj_detect_dev OBJ_DETECT_MYSQL_PWD=obj_detect_dev_pwd OBJ_DETECT_MYSQL_HOST=127.0.0.1 OBJ_DETECT_MYSQL_DB=obj_detect_dev_db python3 -m src.app
```
## Running unit tests (runs with SQL Lite Database):  
```
OBJ_DETECT_ENV=test OBJ_DETECT_MYSQL_DB=obj_detect_test_db python3 -m unittestst
```

## API Usage  
Here we demonstrate how to send requests to the api and receive responses

### Insert model name:
```
curl -s -H "Content-Type: application/json" -X POST -d '{"name":"resnet", "description": "resnet50 model as a feature extractor"}' http://localhost:5000/api/v1/models  

{
"id": "d0706b53-6e4d-4970-ae1e-0db065da0758",
"name": "resnet",
"description": "resnet50 model as a feature extractor"
}

```  

### Train model
```
curl -s -H "Content-Type: application/json" -X POST -d '{"mode":"few_shot", "models":{"segmentation":"7020c689-282a-4d35-bbb4-ee51c94613cd", "feature_extraction":"d0706b53-6e4d-4970-ae1e-0db065da0758", "classification":"36e76e67-9b68-48c2-a043-28de1f75467f"}, "n_per_label_train":"20", "n_per_label_test":"5", "labels":["cat", "dog", "car", "phone", "person"]}' http://localhost:5000/api/v1/train




```  

