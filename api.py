import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    # call model here
    # return model 
    return "<h1>Model response here.</p>"

app.run()