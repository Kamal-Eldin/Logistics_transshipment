import requests

host= "discharge-service-bean.eba-kfty3ew7.eu-central-1.elasticbeanstalk.com"
local_url= "http://localhost:9696/score"
url= f"http://{host}:80/score"

message = {
    "vesseldwt": 500,
    "n_stevs": 5,
    "process_time": 6,
    "vesseltype_1.0": 0,

    "vesseltype_2.0": 0,
    "vesseltype_3.0": 1,
    "vesseltype_4.0": 0,
    "vesseltype_5.0": 0,

    "traveltype_ARRIVAL": 1,
    "traveltype_SHIFT": 0,
    "bulk_liquid": 1,
    "bulk_solid": 0
}

if __name__=='__main__':
    print(requests.post(url=url, json=message).json())