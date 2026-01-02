import requests, json

response = requests.get("https://www.rolimons.com/itemapi/itemdetails")

def download():
    items = []

    for id, item in response.json()["items"].items():
        print(id, item)

        items.append(id)

    for i in range(0, len(items), 32):
        id_list = []

        for j in range(i, i + 32):
            if j < len(items):
                id_list.append(str(items[j]))

        print(",".join(id_list))
        
        response = requests.get(f"https://thumbnails.roblox.com/v1/assets?assetIds={",".join(id_list)}&size=250x250&format=Png&isCircular=false")

        print(response.json())

        for thumbnail in response.json()["data"]:
            image = requests.get(thumbnail["imageUrl"])

            with open(f"{thumbnail["targetId"]}.png", "wb") as f:
                f.write(image.content)

def gen_list():
    out = []

    for id, item in response.json()["items"].items():
        out.append({
            "id": id,
            "name": item[0]
        })

    with open("generator/assets/db.json", "w") as f:
        f.write(json.dumps(out, separators=(',', ':')))

gen_list()