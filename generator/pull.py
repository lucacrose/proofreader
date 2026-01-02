import requests

cursor = ""

while True: 
    response = requests.get(f"https://catalog.roblox.com/v2/search/items/details?taxonomy=tZsUsd2BqGViQrJ9Vs3Wah&creatorName=Roblox&salesTypeFilter=2&includeNotForSale=true&limit=120&cursor={cursor}")

    data = response.json()

    #print(response.status_code, data, cursor)

    items = response.json()["data"]

    for i in range(0, len(items), 16):
        id_list = []

        for j in range(i, i + 16):
            if j < len(items):
                id_list.append(str(items[j]["id"]))

        print(",".join(id_list))
        
        response = requests.get(f"https://thumbnails.roblox.com/v1/assets?assetIds={",".join(id_list)}&size=250x250&format=Png&isCircular=false")

        print(response.json())

        for thumbnail in response.json()["data"]:
            image = requests.get(thumbnail["imageUrl"])

            with open(f"{thumbnail["targetId"]}.png", "wb") as f:
                f.write(image.content)

    cursor = data["nextPageCursor"]

    if not cursor:
        break
