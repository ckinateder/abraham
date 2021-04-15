# abraham

Algorithmically predict public sentiment on a company using VADER sentiment analysis.

## Sample Output

```python
from starwars import Darth
darthvader = Darth(period='1d', splitting=True)

scores = darthvader.sentiment(["robinhood", 
                      "johnson and johnson", 
                      "bitcoin", 
                      "dogecoin", 
                      "biden",  
                      "amazon"])  # this command takes a bit of time to run because it has to download lots of articles

print(scores)

'''
{'robinhood': 
    {
        'avg': 0.3798676562301132, 
        'nice': 'positive :)'
     },
 'johnson and johnson': 
    {
        'avg': 0.27466788299009787, 
        'nice': 'positive :)'
    },
 'bitcoin': 
    {
        'avg': 0.28669931035859125, 
        'nice': 'positive :)'
    },
 'dogecoin': 
    {
        'avg': 0.2837840361036227, 
        'nice': 'positive :)'
    },
 'biden': 
    {
        'avg': 0.2404157345348728, 
        'nice': 'positive :)'
    },
 'amazon': 
    {
        'avg': 0.2894022880254384, 
        'nice': 'positive :)'
    }
}
'''
```
