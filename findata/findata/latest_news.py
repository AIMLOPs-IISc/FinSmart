from GoogleNews import GoogleNews

class News:
    def __init__(self, lang="en", region="IN", encoding="utf-8", period=30):
        self.googlenews = GoogleNews(lang=lang, region=region)
        self.googlenews.set_encode(encoding)
        self.googlenews.set_period(f'{period}d')

    def get_headlines(self, name, max_count=9, predictor=None):
        self.googlenews.clear()
        out = []
        self.googlenews.get_news(name)
        self.googlenews.search(name)
        results = self.googlenews.results(sort=True)
        count = max_count
        for res in results:
            if count <= 0:
                break
            if len(res['title']) > 30:
                count -= 1
                if res['link']:
                    if not res['link'].startswith("https://"):
                        res['link'] = "https://" + res['link']
                else:
                    res['link'] = "#"
                if res['img']:
                    if not res['img'].startswith("https://"):
                        res['img'] = "https://" + res['img']
                else:
                    res['img'] = "/static/sample_img.jpeg"
                res["srno"] = str(max_count - count)
                res["label"] = "na"
                if predictor:
                    res["label"] = predictor(res["title"])
                out.append(res)
        return out