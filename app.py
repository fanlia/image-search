
from elasticsearch import Elasticsearch
from flask import Flask, jsonify, request
from towhee import pipeline
from numpy import linalg as LA

embedding_pipeline = pipeline('image-embedding')

def extract_feature(img_path):
    feat = embedding_pipeline(img_path)
    norm_feat = feat / LA.norm(feat)
    return norm_feat.tolist()

app = Flask(__name__)
es = Elasticsearch(['elasticsearch'])

index = 'image'
body = {
    'mappings': {
        'properties': {
            'url': {
                'type': 'keyword',
            },
            'vector': {
                'type': 'dense_vector',
                'dims': 2048,
            },
        },
    },
}

if not es.indices.exists(index):
    es.indices.create(index, body)

@app.route('/')
def search():
    url = request.args.get('url')
    if url is None:
        return 'url required', 400

    feature = extract_feature(url)
    body = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "1 / (l2norm(params.queryVector, doc['vector']) + 1)",
                    "params": {
                        "queryVector": feature
                    }
                }
            }
        }
    }
    result = es.search(body, index, _source_excludes=['vector'])
    urls = list(map(lambda x: x['_source']['url'], result['hits']['hits']))

    return jsonify({'result': urls})

@app.route('/create')
def create():
    url = request.args.get('url')
    if url is None:
        return 'url required', 400

    body = {
        "query": {
            "term": {
                "url": url,
            }
        }
    }
    result = es.search(body, index, _source=False)
    hits = result['hits']['hits']

    feature = extract_feature(url)

    if len(hits) > 0:
        id = hits[0]['_id']
        body = { 'doc': { 'vector': feature }}
        result = es.update(index, id, body)
    else:
        document = {
            'url': url,
            'vector': feature,
        }
        result = es.index(index, document=document)
    return jsonify({'result': result['result']})
@app.route('/delete')
def delete():
    url = request.args.get('url')
    if url is None:
        return 'url required', 400

    body = {
        "query": {
            "term": {
                "url": url,
            }
        }
    }
    result = es.search(body, index, _source=False)
    hits = result['hits']['hits']

    feature = extract_feature(url)

    if len(hits) == 0:
        return 'url not found', 400

    id = hits[0]['_id']
    result = es.delete(index, id)
    return jsonify({'result': result['result']})
