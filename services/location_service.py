import requests

def get_states(country_name):
    url = "https://countriesnow.space/api/v0.1/countries/states"
    try:
        response = requests.post(url, json={"country": country_name}, timeout=5)
        data = response.json()
        if not data["error"]:
            return [s["name"] for s in data["data"]["states"]]
    except:
        pass
    return []