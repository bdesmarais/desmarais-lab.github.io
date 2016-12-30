import sys, csv
from flask import Flask, render_template
from flask_flatpages import FlatPages
from flask_frozen import Freezer
from flask_table import Table, Col, LinkCol

DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'

app = Flask(__name__)
app.config.from_object(__name__)
flatpages = FlatPages(app)
freezer = Freezer(app)
app.config['FREEZER_BASE_URL'] = ''
pages = [p for p in flatpages if p.path != 'home']

class PictureCol(Col):
    def td_format(self, content):
        return '<img src=static/' + content + '.jpg>'

class PeopleTable(Table):
    name = Col('name')
    description = Col('description')
    picture = PictureCol('picture')
    email = LinkCol('email')
    link = LinkCol('link')

class ItemTable(Table):
    name = LinkCol('Name', 'single_item',
                   url_kwargs=dict(id='id'), attr='name')
    description = Col('Description')    

class People(object):
    def __init__(self, name, description, picture, email, link):
        self.name = name
        self.description = description
        self.picture = picture
        self.email = email
        self.link = link


items = pd.read_csv('static/people.csv')
table = PeopleTable(items.to_dict('records'))
    
@app.route('/')
def index():
    return render_template('index.html', homepage=flatpages.get('home'), pages=pages)

@app.route('/people/')
def people():
    return render_template('people.html', page=flatpages.get('people'), table=table)

@app.route('/<path:path>/')
def page(path):
    page = flatpages.get_or_404(path)
    return render_template('page.html', page=page)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        freezer.freeze()
    else:
        app.run(port=8000)
