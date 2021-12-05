from flask import Flask, render_template,request
from twitterapi import main
app = Flask(__name__)


# GLOBAL VARIABLES
CURR_STOCK = ""

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/getdata',methods=['POST','GET'])
def getdata():
    global INPUT
    INPUT = str(request.form.get('company_name'))
    a = main(INPUT)
    prediction = []
    prediction.append(a['positive'])
    prediction.append(a['negative'])
    prediction.append(a['nuetral'])    
    return render_template("index.html", prediction = prediction)

if __name__ == "__main__":
    app.run()