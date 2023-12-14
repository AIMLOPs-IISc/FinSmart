from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from pydantic import BaseModel

from findata import NSE, News
from finbert import distil_bert
from xgbpred import xgb_stock
from t5summ import t5_summarize

import prometheus_client as prom

r2_metric = prom.Gauge('xgbstockpred_r2_score', 'R2 score for random stock prediction')
corr_metric = prom.Gauge('finbert_accuracy', "Accuracy of finbert model.")
rouge1_metric = prom.Gauge('summary_precision', "Precision of summarizer module")

app = FastAPI()
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

print("initializing NSE")
_nse = NSE()
_nse.datapath = "./nsedata"
print("Initializing News")
_news = News()
print("Initializing Sentiment analysis")
senti_class = distil_bert.FinSentiment()
print("Initializing XGboost Prediction")
_pred = xgb_stock.Predictor()
print("Initializing T5 Summarizer")
_summ = t5_summarize.FinSumm()
print("Initialization Process Complete")


async def update_metrics():
    r2 = _pred.evaluate()
    acc = senti_class.metric(5)
    sm = _summ.metric()
    r2_metric.set(r2)
    corr_metric.set(acc)
    rouge1_metric.set(sm)


@app.get("/metrics")
async def get_metrics():
    await update_metrics()
    return Response(media_type="text/plain", content=prom.generate_latest())


async def get_mood(news):
    mood = 0
    count = 0
    label2id = {"Positive": 1, "Neutral": 0, "Negative": -1}
    for ne in news:
        if ne["label"] in label2id:
            mood += label2id[ne["label"]]
            count = count + 1
    if count == 0:
        return 0
    return mood / count


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/overview", response_class=HTMLResponse)
async def get_overview(request: Request):
    print("called")
    hl = _nse.highlights(5)
    print("data_return")
    return templates.TemplateResponse("overview.html", {"request": request, "highlights": hl})


@app.get("/prediction", response_class=HTMLResponse)
async def analysis_page(request: Request):
    name = "Business"
    news = _news.get_headlines("Business", predictor=senti_class.predict)

    mood = await get_mood(news)

    return templates.TemplateResponse("analysis.html", {"request": request, "name": name, "news": news, "mood": mood})


@app.get("/team", response_class=HTMLResponse)
async def read_teams(request: Request):
    return templates.TemplateResponse("team.html", {"request": request})


@app.get("/shorten", response_class=HTMLResponse)
async def read_teams(request: Request):
    return templates.TemplateResponse("shorten.html", {"request": request})


@app.get("/prediction/api/tickers", response_class=JSONResponse)
async def get_ticker_suggestion(request: Request):
    out = {k: None for k in _nse.tickers()}
    content = jsonable_encoder(out)
    return JSONResponse(content=content)


@app.get("/prediction/api/news/{symbol}", response_class=JSONResponse)
async def read_item(request: Request, symbol: str):
    name = _nse.company(symbol)
    if name != "na":
        news = _news.get_headlines(name, 9, senti_class.predict)
    else:
        name = symbol
        news = [
            {
                'srno': "1",
                'title': "Unable to find news related requested symbol. Please verify if this symbol is listed in NSE.",
                'date': "No News Found.",
                'media': "NSE",
                'link': "https://www.nseindia.com/",
                'label': "na"
            }
        ]
    mood = await get_mood(news)
    res = {
        "name": name,
        "mood": mood,
        "news": news
    }
    content = jsonable_encoder(res)
    return JSONResponse(content=content)


@app.get("/prediction/api/projection/{symbol}", response_class=JSONResponse)
async def get_ticker_suggestion(request: Request, symbol: str):
    data = _nse.historical(symbol.lower(), 100)
    score = 0
    if len(data) != 0:
        data = data[["CH_TIMESTAMP", "CH_TRADE_HIGH_PRICE",
                     "CH_TRADE_LOW_PRICE",
                     "CH_OPENING_PRICE",
                     "CH_CLOSING_PRICE",
                     "CH_TOT_TRADED_QTY"]]
        data.columns = ["Date", "High", "Low", "Open", "Close", "Volume"]
        out, score = _pred.next_n_pred(data, days=10)
        data = []
        colors = ["green", "blue", "red", "cyan"]
        for itr, metric in enumerate(["High", "Low", "Open", "Close"]):
            data.append({
                "x":out["Date"].values.tolist()[:-10],
                "y":out[f"True_{metric}"].values.tolist()[:-10],
                "mode": 'markers',
                "name": f"True_{metric}",
                "marker":{"color":colors[itr]}
            })
            data.append({
                "x": out["Date"].values.tolist(),
                "y": out[f"Pred_{metric}"].values.tolist(),
                "mode": 'lines',
                "name": f"Pred_{metric}",
                "line":{"dash":'dot',"width": 1, "color":colors[itr]}
            })
    content = jsonable_encoder({"data":data, "score": score})
    return JSONResponse(content=content)


class InpStr(BaseModel):
    text: str

@app.post("/shorten/api/summarize", response_class=JSONResponse)
async def get_summary(input_txt: InpStr):
    data = _summ.generate_summary(input_txt.text)
    content = jsonable_encoder({"summary": data})
    return JSONResponse(content=content)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
