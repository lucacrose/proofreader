([newItems, is_empty_trade, backgrounds_count, config]) => {
    function generateRandomUsername() {
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
        const length = Math.floor(Math.random() * (20 - 3 + 1)) + 3;
        
        let result = "";
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }

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

    function randomAlphanumericWithSpaces(min = 2, max = 48) {
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        const length = Math.floor(Math.random() * (max - min + 1)) + min;

        let result = chars[Math.floor(Math.random() * chars.length)];

        for (let i = 1; i < length; i++) {
            if (!config.name.double_spacing && result[result.length - 1] === " ") {
                result += chars[Math.floor(Math.random() * chars.length)];
            } else {
            result += Math.random() < config.name.space_chance
                ? " "
                : chars[Math.floor(Math.random() * chars.length)];
            }
        }

        return result;
    }

    const randomIndex = getRandomInt(0, backgrounds_count - 1);
    const randomBgUrl = `url('../backgrounds/background_${randomIndex}.jpg')`;

    if (Math.random() < config.background.chance) {
        const container = document.querySelector(".container-main");
        container.style.backgroundImage = randomBgUrl;
        container.style.backgroundSize = "cover";
        container.style.backgroundPosition = "center";
        container.style.backgroundRepeat = "no-repeat";
    }

    if (Math.random() < config.recolor.chance) {
        const content = document.querySelector(".content");
        content.style.backgroundColor = getRandomColor(config.recolor.container_min_opacity, config.recolor.container_max_opacity);
        content.style.color = getRandomColor(config.recolor.text_min_opacity, config.recolor.text_max_opacity);
    }
    
    const nameElements = document.querySelectorAll('.paired-name .element');
    nameElements[0].innerText = generateRandomUsername();
    nameElements[1].innerText = generateRandomUsername();
            
    const cards = document.querySelectorAll('div[trade-item-card]');
    cards.forEach((card, index) => {
        const isFirstSide = index < 4;
        const sideIndex = isFirstSide ? 0 : 1;
        const itemIndex = isFirstSide ? index : index - 4;
        const data = newItems[sideIndex][itemIndex];
        const hideName = Math.random() < config.cards.name_hide_chance;
        const hideThumb = Math.random() < config.cards.thumb_hide_chance;

        if (data) {
            card.style.visibility = "visible";
            card.style.opacity = "1";
            card.setAttribute("data-item-id", data.id);

            const img = card.querySelector('img');
            img.src = data;

            const priceLabel = card.querySelector('.text-robux');
            priceLabel.innerText = getRandomNumberEqualDigits(1, 9).toLocaleString();

            const priceLine = priceLabel.closest('.item-card-price');

            if (
                priceLine &&
                Math.random() < config.cards.duplicate_price_line_chance &&
                !priceLine.nextElementSibling?.classList.contains('item-card-price')
            ) {
                const clone = priceLine.cloneNode(true);

                const cloneValue = clone.querySelector('.text-robux');
                cloneValue.innerText = getRandomNumberEqualDigits(1, 9).toLocaleString();

                priceLine.parentElement.insertBefore(clone, priceLine.nextSibling);
            }

            const nameLabel = card.querySelector('.item-card-name');
            nameLabel.innerText = randomAlphanumericWithSpaces();
            nameLabel.style.lineHeight = `${getRandomInt(config.cards.line_height_min, config.line_height_max)}px`;

            if (hideName) nameLabel.style.display = "none";
            if (hideThumb) img.parentElement.parentElement.parentElement.parentElement.style.display = "none";
        } else {
            card.style.opacity = "0"; 
            card.setAttribute("data-item-id", "");
        }
    });
    
    const robuxLines = document.querySelectorAll(".robux-line");
    robuxLines[0].style.display = is_empty_trade ? "none" : (Math.random() < config.robux_lines.hide_chance ? "none" : "");
    robuxLines[2].style.display = is_empty_trade ? "none" : (Math.random() < config.robux_lines.hide_chance ? "none" : "");
            
    document.querySelectorAll(".robux-line-value").forEach(el => {
        el.textContent = getRandomNumberEqualDigits(1, 10).toLocaleString();
    });

    document.querySelectorAll(".limited-icon-container").forEach(container => {
        const numberContainer = container.querySelector(".limited-number-container");
        const numberSpan = container.querySelector(".limited-number");
        if (!numberContainer || !numberSpan) return;
        const show = Math.random() < config.cards.display_serial_chance;
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
        const offsetLeft = Math.floor(getRandomInt(config.cards.left_offset_min, config.cards.left_offset_max))
        const offsetTop = Math.floor(getRandomInt(config.cards.top_offset_min, config.cards.top_offset_max));
        el.style.marginLeft = `${offsetLeft}px`;
        el.style.marginTop = `${offsetTop}px`;
    });
    
    const withColon = text =>
    Math.random() < config.robux_lines.colon_suffix_chance ? `${text}:` : text;

    document.querySelectorAll('.trade-list-detail-offer').forEach(offer => {
        if (Math.random() > config.robux_lines.duplicate_line_chance) return;

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
