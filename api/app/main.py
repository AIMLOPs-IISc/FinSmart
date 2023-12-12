import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from findata import NSE, News
from finbert import distil_bert


app = FastAPI()
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

_nse = NSE()
_news = News()
senti_class = distil_bert.FinSentiment()

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
async def get_overview(request:Request):
    hl = _nse.highlights(5)
    return templates.TemplateResponse("overview.html", {"request": request, "highlights":hl})

@app.get("/prediction", response_class=HTMLResponse)
async def analysis_page(request: Request):
    name = "Business"
    news = _news.get_headlines("Business", predictor=senti_class.predict)

    mood = await get_mood(news)

    return templates.TemplateResponse("analysis.html", {"request": request, "name":name, "news": news , "mood":mood})

@app.get("/team", response_class=HTMLResponse)
async def read_teams(request: Request):
    return templates.TemplateResponse("team.html", {"request": request})


@app.get("/prediction/api/tickers",response_class=JSONResponse)
async def get_ticker_suggestion(request: Request):
    out = {k:None for k in _nse.tickers()}
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
                'srno':"1",
                'title': "Unable to find news related requested symbol. Please verify if this symbol is listed in NSE.",
                'date': "No News Found.",
                'media': "NSE",
                'link': "https://www.nseindia.com/",
                'label':"na"
            }
        ]

    mood = await get_mood(news)

    res = {
        "name": name,
        "mood":mood,
        "news": news
    }
    print(res)
    content = jsonable_encoder(res)
    return JSONResponse(content=content)


@app.get("/prediction/api/projection/{symbol}",response_class=JSONResponse)
async def get_ticker_suggestion(request: Request, symbol: str):
    out = {k:None for k in nse.tickers()}
    content = jsonable_encoder(out)
    return JSONResponse(content=content)

@app.get("/prediction/api/details/{symbol}",response_class=JSONResponse)
async def get_ticker_suggestion(request: Request, symbol: str):
    out = {

    }
    content = jsonable_encoder(out)
    return JSONResponse(content=content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)