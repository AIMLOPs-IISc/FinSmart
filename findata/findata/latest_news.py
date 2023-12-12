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
                temp = {}
                count -= 1

                if res['link']:
                    if not res['link'].startswith("https://"):
                        temp['link'] = "https://" + res['link']
                    else:
                        temp['link'] = res['link']
                else:
                    temp['link'] = "#"

                if res['img']:
                    if not res['img'].startswith("https://"):
                        temp['img'] = "https://" + res['img']
                    else:
                        temp['img'] = res['img']
                else:
                    temp['img'] = "/static/sample_img.jpeg"

                if res['date']:
                    temp['date'] = str(res['date'])
                else:
                    temp['date'] = "Date Not Specified"

                if res['media']:
                    temp['media'] = str(res['media'])
                else:
                    temp['media'] = "FinSmart"

                temp["srno"] = str(max_count - count)
                temp["label"] = "n/a"
                if predictor:
                    temp["label"] = predictor(res["title"])

                out.append(temp)
        return out