([newItems, userData, is_empty_trade, backgrounds_count]) => {
    function getRandomInt(min, max) {
        min = Math.ceil(min);
        max = Math.floor(max);
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }
    
    function getRandomNumberEqualDigits(minDigits = 1, maxDigits = 5) {
        const digits = getRandomInt(minDigits, maxDigits);
        const min = digits === 1 ? 0 : Math.pow(10, digits - 1);
        const max = Math.pow(10, digits) - 1;
        return getRandomInt(min, max);
    }
    
    function getRandomColor(alphaMin = 0.3, alphaMax = 1) {
        const r = Math.floor(Math.random() * 256);
        const g = Math.floor(Math.random() * 256);
        const b = Math.floor(Math.random() * 256);
        const a = (Math.random() * (alphaMax - alphaMin) + alphaMin).toFixed(2);
        return `rgba(${r}, ${g}, ${b}, ${a})`;
    }

    const randomIndex = Math.floor(Math.random() * backgrounds_count) + 1;

    const paddedIndex = String(randomIndex).padStart(3, '0');

    const randomBgUrl = `url('../backgrounds/unsplash_${paddedIndex}.jpg')`;

    if (Math.random() < 0.75) {
        const container = document.querySelector(".container-main");
        container.style.backgroundImage = randomBgUrl;
        container.style.backgroundSize = "cover";
        container.style.backgroundPosition = "center";
        container.style.backgroundRepeat = "no-repeat";
    }

    if (Math.random() < 0.75) {
        const content = document.querySelector(".content");
        content.style.backgroundColor = getRandomColor();
        content.style.color = getRandomColor(0.95, 1);
    }
    
    const nameElements = document.querySelectorAll('.paired-name .element');
    if(nameElements.length >= 2) {
        nameElements[0].innerText = userData.display;
        nameElements[1].innerText = userData.handle;
    }
            
    function randomAlphanumericWithSpaces(min = 2, max = 48) {
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        const length = Math.floor(Math.random() * (max - min + 1)) + min;

        let result = chars[Math.floor(Math.random() * chars.length)];

        for (let i = 1; i < length; i++) {
            if (result[result.length - 1] === " ") {
            result += chars[Math.floor(Math.random() * chars.length)];
            } else {
            result += Math.random() < 0.15
                ? " "
                : chars[Math.floor(Math.random() * chars.length)];
            }
        }

        return result;
    }
            
    const cards = document.querySelectorAll('div[trade-item-card]');
    cards.forEach((card, index) => {
        const isFirstSide = index < 4;
        const sideIndex = isFirstSide ? 0 : 1;
        const itemIndex = isFirstSide ? index : index - 4;
        const data = newItems[sideIndex][itemIndex];
        const hideName = Math.random() < 0.2;
        const hideThumb = Math.random() < 0.2;

        if (data) {
            card.style.visibility = "visible";
            card.style.opacity = "1";
            card.setAttribute("data-item-id", data.id);

            const img = card.querySelector('img');
            if (img) img.src = data.img;

            const priceLabel = card.querySelector('.text-robux');
            if (priceLabel) {
                priceLabel.innerText = getRandomNumberEqualDigits(1, 9).toLocaleString();

                const priceLine = priceLabel.closest('.item-card-price');

                if (
                    priceLine &&
                    Math.random() < 0.35 &&
                    !priceLine.nextElementSibling?.classList.contains('item-card-price')
                ) {
                    const clone = priceLine.cloneNode(true);

                    const cloneValue = clone.querySelector('.text-robux');
                    if (cloneValue) {
                        const base = Number(priceLabel.innerText.replace(/,/g, ''));
                        cloneValue.innerText = getRandomNumberEqualDigits(1, 9).toLocaleString();
                    }

                    priceLine.parentElement.insertBefore(clone, priceLine.nextSibling);
                }
            }

            const nameLabel = card.querySelector('.item-card-name');
            if (nameLabel) {
                nameLabel.innerText = randomAlphanumericWithSpaces();
                nameLabel.style.lineHeight = `${Math.floor(Math.random() * 17) + 12}px`;
            }

            if (hideName && nameLabel) nameLabel.style.display = "none";
            if (hideThumb && img) img.parentElement.parentElement.parentElement.parentElement.style.display = "none";
        } else {
            card.style.opacity = "0"; 
            card.setAttribute("data-item-id", "");
        }
    });
    
    const robuxLines = document.querySelectorAll(".robux-line");
    if(robuxLines.length >= 3) {
        robuxLines[0].style.display = is_empty_trade ? "none" : (Math.random() < 0.25 ? "none" : "");
        robuxLines[2].style.display = is_empty_trade ? "none" : (Math.random() < 0.25 ? "none" : "");
    }
            
    document.querySelectorAll(".robux-line-value").forEach(el => {
        el.textContent = getRandomNumberEqualDigits(1, 10).toLocaleString();
    });

    document.querySelectorAll(".limited-icon-container").forEach(container => {
        const numberContainer = container.querySelector(".limited-number-container");
        const numberSpan = container.querySelector(".limited-number");
        if (!numberContainer || !numberSpan) return;
        const show = Math.random() < 0.5;
        if (show) {
            numberContainer.style.display = "";
            numberSpan.style.display = "";
            numberSpan.textContent = getRandomNumberEqualDigits(1, 7);
        } else {
            numberContainer.style.display = "none";
            numberSpan.style.display = "none";
        }
    });
            
    document.querySelectorAll('.item-card-name, .item-card-price').forEach(el => {
        const offsetLeft = Math.floor(Math.random() * 25)
        const offsetTop = Math.floor(Math.random() * 13);
        el.style.marginLeft = `${offsetLeft}px`;
        el.style.marginTop = `${offsetTop}px`;
    });
    
    const withColon = text =>
    Math.random() < 0.5 ? `${text}:` : text;

    document.querySelectorAll('.trade-list-detail-offer').forEach(offer => {
        if (Math.random() > 0.3) return;

        if (offer.querySelector('.robux-line.total-rap')) return;

        const robuxLines = [...offer.querySelectorAll('.robux-line')];

        const totalValueLine = robuxLines.find(line =>
            line.querySelector('.text-lead')?.textContent
            .replace(/:/g, '')
            .trim() === 'Total Value'
        );

        if (!totalValueLine) return;

        const valueLabel = totalValueLine.querySelector('.text-lead');
        if (valueLabel) {
            valueLabel.textContent = withColon('Total Value');
        }

        const rapValue = [...offer.querySelectorAll('.item-card-price .text-robux')]
            .reduce((sum, el) => sum + Number(el.textContent.replace(/,/g, '')), 0)
            .toLocaleString();
        
        const rapLine = document.createElement('div');
        rapLine.className = 'robux-line total-rap';
        rapLine.innerHTML = `
            <span class="text-lead">${withColon('Total RAP')}</span>
            <span class="robux-line-amount">
            <span class="icon-robux-16x16"></span>
            <span class="text-robux-lg robux-line-value">${rapValue}</span>
            </span>
        `;

        totalValueLine.parentElement.insertBefore(rapLine, totalValueLine);
    });
}
