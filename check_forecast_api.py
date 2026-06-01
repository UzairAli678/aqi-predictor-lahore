import requests
try:
    r = requests.get('http://127.0.0.1:5000/api/forecast', timeout=10)
    print('status', r.status_code)
    try:
        j = r.json()
        print('keys:', list(j.keys()))
        if 'forecast' in j:
            print('forecast len:', len(j['forecast']))
            print('first:', j['forecast'][0])
        else:
            print('response:', j)
    except Exception as e:
        print('json decode error', e)
except Exception as e:
    print('request error', e)
