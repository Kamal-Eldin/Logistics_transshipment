
from flask import Flask

app = Flask("ping_service")

@app.route("/ping", methods=['GET'])
def ping():
    return "PONG\n"


if __name__ == '__main__':  
    app.run(host= '0.0.0.0', port= 9696, debug= True)





