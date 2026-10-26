import requests,json
r=requests.get('https://api.github.com/search/repositories',params={'q':'smoking detection','sort':'stars','per_page':5},timeout=30,headers={'Accept':'application/vnd.github+json','User-Agent':'dms-research'})
print(r.status_code)
print(r.text[:1500])
print('rate',r.headers.get('X-RateLimit-Remaining'))
