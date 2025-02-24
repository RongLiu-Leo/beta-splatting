/*
 * Dics: Definitive image comparison slider. A multiple image vanilla comparison slider.
 *
 * By Abel Cabeza Román, a Codictados developer
 * Src: https://github.com/abelcabezaroman/definitive-image-comparison-slider
 * Example: http://codictados.com/portfolio/definitive-image-comparison-slider-demo/
 */

/**
 *
 */

/**
 *
 * @type {{container: null, filters: null, hideTexts: null, textPosition: string, linesOrientation: string, rotate: number, arrayBackgroundColorText: null, arrayColorText: null, linesColor: null}}
 */
let defaultOptions = {
    container: null, // **REQUIRED**: HTML container | `document.querySelector('.b-dics')` |
    filters: null, // Array of CSS string filters  |`['blur(3px)', 'grayscale(1)', 'sepia(1)', 'saturate(3)']` |
    hideTexts: true, // Show text only when you hover the image container |`true`,`false`|
    textPosition: "center", // Set the prefer text position  |`'center'`,`'top'`, `'right'`, `'bottom'`, `'left'` |
    linesOrientation: "horizontal", // Change the orientation of lines  |`'horizontal'`,`'vertical'` |
    rotate: 0, // Rotate the image container (not too useful but it's a beatiful effect. String of rotate CSS rule)  |`'45deg'`|
    arrayBackgroundColorText: null, // Change the bacground-color of sections texts with an array |`['#000000', '#FFFFFF']`|
    arrayColorText: null, // Change the color of texts with an array  |`['#FFFFFF', '#000000']`|
    linesColor: null // Change the lines and arrows color  |`'rgb(0,0,0)'`|
  
  };
  
  /**
   *
   * @param options
   * @constructor
   */
  let Dics = function(options) {
    this.options = utils.extend({}, [defaultOptions, options], {
      clearEmpty: true
    }); 
  
    this.container = this.options.container;
    console.warn(this.container);
    if (this.container == null) {
      console.error("Container element not found!");
    } else {
  
      this._setOrientation(this.options.linesOrientation, this.container);
      this.medias = this._getMedias();
      this.sliders = [];
      this._activeSlider = null;
  
  
      this._load(this.medias[0]);
  
    }
  };
  
  
  /**
   *
   * @private
   */
  Dics.prototype._load = function(firstMedia, maxCounter = 100000) {
    if (firstMedia.tagName === 'IMG') {
        if (firstMedia.naturalWidth) {
            this._buidAfterFirstMediaLoad(firstMedia);
            window.addEventListener("resize", () => {
                this._setContainerWidth(firstMedia);
                this._resetSizes();
            });
        } else {
            if (maxCounter > 0) {
                maxCounter--;
                setTimeout(() => {
                    this._load(firstMedia, maxCounter);
                }, 100);
            } else {
                console.error("Error loading medias");
            }
        }
    } else if (firstMedia.tagName === 'VIDEO') {
        firstMedia.addEventListener('loadedmetadata', () => {
            this._buidAfterFirstMediaLoad(firstMedia);
            window.addEventListener("resize", () => {
                this._setContainerWidth(firstMedia);
                this._resetSizes();
            });
        });

        // Ensure the video is loaded
        if (firstMedia.readyState >= 1) {
            this._buidAfterFirstMediaLoad(firstMedia);
        } else {
            firstMedia.load(); // Trigger loading if not loaded
        }
    }
};
  
  
  /**
   *
   * @private
   */
  Dics.prototype._buidAfterFirstMediaLoad = function(firstMedia) {
    this._setContainerWidth(firstMedia);
  
    this._build();
    this._setEvents();
  };
  
  
  /**
   *
   * @private
   */
  Dics.prototype._setContainerWidth = function(firstMedia) {
    this.options.container.style.height = `${this._calcContainerHeight(firstMedia)}px`;
  };
  
  
  /**
   *
   * @private
   */
  Dics.prototype._setOpacityContainerForLoading = function(opacity) {
    this.options.container.style.opacity = opacity;
  };
  
  
  /**
   * Reset sizes on window size change
   * @private
   */
  Dics.prototype._resetSizes = function() {
    let dics = this;
    let mediasLength = dics.medias.length;
  
    let initialMediasContainerWidth = dics.container.getBoundingClientRect()[dics.config.sizeField] / mediasLength;
  
    const sections$$ = dics.container.querySelectorAll("[data-function='b-dics__section']");
    for (let i = 0; i < sections$$.length; i++) {
      let section$$ = sections$$[i];
  
      section$$.style.flex = `0 0 ${initialMediasContainerWidth}px`;
  
      section$$.querySelector(".b-dics__media").style[this.config.positionField] = `${i * -initialMediasContainerWidth}px`;
  
      const slider$$ = section$$.querySelector(".b-dics__slider");
      if (slider$$) {
        slider$$.style[this.config.positionField] = `${initialMediasContainerWidth * (i + 1)}px`;
  
      }
  
    }
  
  };
  
  /**
   * Build HTML
   * @private
   */
  Dics.prototype._build = function() {
    let dics = this;

    dics._applyGlobalClass(dics.options);

    let mediaLength = dics.medias.length;  // Supports both images and videos
    let initialMediaContainerWidth = dics.container.getBoundingClientRect()[dics.config.sizeField] / mediaLength;

    for (let i = 0; i < mediaLength; i++) {
        let media = dics.medias[i];  // Now supports both <img> and <video>
        let section = dics._createElement("div", "b-dics__section");
        let mediaContainer = dics._createElement("div", "b-dics__media-container");
        let slider = dics._createSlider(i, initialMediaContainerWidth);

        dics._createAltText(media, i, mediaContainer);
        dics._applyFilter(media, i, dics.options.filters);
        dics._rotate(media, mediaContainer);

        section.setAttribute("data-function", "b-dics__section");
        section.style.flex = `0 0 ${initialMediaContainerWidth}px`;

        media.classList.add("b-dics__media");  // Apply class to both <img> and <video>

        section.appendChild(mediaContainer);
        mediaContainer.appendChild(media);

        if (i < mediaLength - 1) {
            section.appendChild(slider);
        }

        dics.container.appendChild(section);

        media.style[dics.config.positionField] = `${i * -initialMediaContainerWidth}px`;
    }

    this.sections = this._getSections();
    this._setOpacityContainerForLoading(1);
  };
  
  
  /**
   *
   * @returns {NodeListOf<SVGElementTagNameMap[string]> | NodeListOf<HTMLElementTagNameMap[string]> | NodeListOf<Element>}
   * @private
   */
  Dics.prototype._getMedias = function() {
    let mediaElements = Array.from(this.container.querySelectorAll("img, video"));

    if (mediaElements.length === 0) {
        console.warn("No images or videos found in the container.");
    }
    const videos = mediaElements.filter(el => el.tagName.toLowerCase() === "video");
    let readyCount = 0;

    videos.forEach(video => {
      // Add click listener to toggle play/pause for all videos in this container
      video.addEventListener('click', () => {
        // Use the clicked video's state to determine the action
        const shouldPlay = video.paused;
        videos.forEach(v => {
          // If playing is desired, start them; otherwise, pause them.
          if (shouldPlay) {
            v.play();
          } else {
            v.pause();
          }
        });
      });

      // Wait for metadata to load before starting
      video.addEventListener('loadedmetadata', () => {
        readyCount++;
        if (readyCount === videos.length) {
          // Set initial time to 0
          videos.forEach(v => {
            v.currentTime = 0;
          });
          // Start synchronizing the videos within this container
          syncVideos(videos);
        }
      });
    });
  
    // Function to keep videos in the same container in sync
    function syncVideos(videos) {
      const masterVideo = videos[0]; // Choose the first video as the master
      masterVideo.addEventListener('timeupdate', () => {
        const masterTime = masterVideo.currentTime;
        videos.forEach(video => {
          // Adjust only if the time difference is significant
          if (video !== masterVideo && Math.abs(video.currentTime - masterTime) > 0.1) {
            video.currentTime = masterTime;
          }
        });
      });
    }

    return mediaElements;
};
  
  
  /**
   *
   * @returns {NodeListOf<SVGElementTagNameMap[string]> | NodeListOf<HTMLElementTagNameMap[string]> | NodeListOf<Element>}
   * @private
   */
  Dics.prototype._getSections = function() {
    return this.container.querySelectorAll("[data-function=\"b-dics__section\"]");
  };
  
  /**
   *
   * @param elementClass
   * @param className
   * @returns {HTMLElement | HTMLSelectElement | HTMLLegendElement | HTMLTableCaptionElement | HTMLTextAreaElement | HTMLModElement | HTMLHRElement | HTMLOutputElement | HTMLPreElement | HTMLEmbedElement | HTMLCanvasElement | HTMLFrameSetElement | HTMLMarqueeElement | HTMLScriptElement | HTMLInputElement | HTMLUnknownElement | HTMLMetaElement | HTMLStyleElement | HTMLObjectElement | HTMLTemplateElement | HTMLBRElement | HTMLAudioElement | HTMLIFrameElement | HTMLMapElement | HTMLTableElement | HTMLAnchorElement | HTMLMenuElement | HTMLPictureElement | HTMLParagraphElement | HTMLTableDataCellElement | HTMLTableSectionElement | HTMLQuoteElement | HTMLTableHeaderCellElement | HTMLProgressElement | HTMLLIElement | HTMLTableRowElement | HTMLFontElement | HTMLSpanElement | HTMLTableColElement | HTMLOptGroupElement | HTMLDataElement | HTMLDListElement | HTMLFieldSetElement | HTMLSourceElement | HTMLBodyElement | HTMLDirectoryElement | HTMLDivElement | HTMLUListElement | HTMLHtmlElement | HTMLAreaElement | HTMLMeterElement | HTMLAppletElement | HTMLFrameElement | HTMLOptionElement | HTMLImageElement | HTMLVideoElement | HTMLLinkElement | HTMLHeadingElement | HTMLSlotElement | HTMLVideoElement | HTMLBaseFontElement | HTMLTitleElement | HTMLButtonElement | HTMLHeadElement | HTMLParamElement | HTMLTrackElement | HTMLOListElement | HTMLDataListElement | HTMLLabelElement | HTMLFormElement | HTMLTimeElement | HTMLBaseElement}
   * @private
   */
  Dics.prototype._createElement = function(elementClass, className) {
    let newElement = document.createElement(elementClass);
  
    newElement.classList.add(className);
  
    return newElement;
  };
  
  /**
   * Set need DOM events
   * @private
   */
  Dics.prototype._setEvents = function() {
    let dics = this;
  
    dics._disableMediaDrag();
  
    dics._isGoingRight = null;
  
    let oldx = 0;
  
    let listener = function(event) {
  
      let xPageCoord = event.pageX ? event.pageX : event.touches[0].pageX;
  
      if (xPageCoord < oldx) {
        dics._isGoingRight = false;
      } else if (xPageCoord > oldx) {
        dics._isGoingRight = true;
      }
  
      oldx = xPageCoord;
  
      let position = dics._calcPosition(event);
  
      let beforeSectionsWidth = dics._beforeSectionsWidth(dics.sections, dics.medias, dics._activeSlider);
  
      let calcMovePixels = position - beforeSectionsWidth;
  
      dics.sliders[dics._activeSlider].style[dics.config.positionField] = `${position}px`;
  
      dics._pushSections(calcMovePixels, position);
    };
  
    dics.container.addEventListener("click", function(event) {
      let el = event.target;
      let sliderFound = false;
      while (el && el !== dics.container) {
        if (el.classList && el.classList.contains("b-dics__slider")) {
          sliderFound = true;
          break;
        }
        el = el.parentElement;
      }
      if (sliderFound) {
        listener(event);
      }
    });
  
    for (let i = 0; i < dics.sliders.length; i++) {
      let slider = dics.sliders[i];
      utils.setMultiEvents(slider, ["mousedown", "touchstart"], function(event) {
        dics._activeSlider = i;
  
        dics._clickPosition = dics._calcPosition(event);
  
        slider.classList.add("b-dics__slider--active");
  
        utils.setMultiEvents(dics.container, ["mousemove", "touchmove"], listener);
      });
    }
  
  
    let listener2 = function() {
      let activeElements = dics.container.querySelectorAll(".b-dics__slider--active");
  
      for (let activeElement of activeElements) {
        activeElement.classList.remove("b-dics__slider--active");
        utils.removeMultiEvents(dics.container, ["mousemove", "touchmove"], listener);
      }
    };
  
    utils.setMultiEvents(document.body, ["mouseup", "touchend"], listener2);
  
  
  };
  
  /**
   *
   * @param sections
   * @param medias
   * @param activeSlider
   * @returns {number}
   * @private
   */
  Dics.prototype._beforeSectionsWidth = function(sections, medias, activeSlider) {
    let width = 0;
    for (let i = 0; i < sections.length; i++) {
      let section = sections[i];
      if (i !== activeSlider) {
        width += section.getBoundingClientRect()[this.config.sizeField];
      } else {
        return width;
      }
    }
  };
  
  /**
   *
   * @returns {number}
   * @private
   */
  Dics.prototype._calcContainerHeight = function(firstMedia) {
    let mediaHeight, mediaWidth;
  
    if (firstMedia.tagName === "IMG") {
        mediaHeight = firstMedia.naturalHeight;
        mediaWidth = firstMedia.naturalWidth;
    } else if (firstMedia.tagName === "VIDEO") {
        mediaHeight = firstMedia.videoHeight;
        mediaWidth = firstMedia.videoWidth;
    } else {
        console.error("Unsupported media type:", firstMedia);
        return 0;
    }

    let containerWidth = this.options.container.getBoundingClientRect().width;

    return (containerWidth / mediaWidth) * mediaHeight;
  };

  
  
  /**
   *
   * @param sections
   * @param medias
   * @private
   */
  Dics.prototype._setLeftToMedias = function(sections, medias) {
    let size = 0;
    for (let i = 0; i < medias.length; i++) {
      let media = medias[i];
  
      media.style[this.config.positionField] = `-${size}px`;
      size += sections[i].getBoundingClientRect()[this.config.sizeField];
  
      this.sliders[i].style[this.config.positionField] = `${size}px`;
  
    }
  };
  
  
  /**
   *
   * @private
   */
  Dics.prototype._disableMediaDrag = function() {
    for (let i = 0; i < this.medias.length; i++) {
      this.sliders[i].addEventListener("dragstart", function(e) {
        e.preventDefault();
      });
      this.medias[i].addEventListener("dragstart", function(e) {
        e.preventDefault();
      });
    }
  };
  
  /**
   *
   * @param media
   * @param index
   * @param filters
   * @private
   */
  Dics.prototype._applyFilter = function(media, index, filters) {
    if (filters) {
      media.style.filter = filters[index];
    }
  };
  
  /**
   *
   * @param options
   * @private
   */
  Dics.prototype._applyGlobalClass = function(options) {
    let container = options.container;
  
  
    if (options.hideTexts) {
      container.classList.add("b-dics--hide-texts");
    }
  
    if (options.linesOrientation === "vertical") {
      container.classList.add("b-dics--vertical");
    }
  
    if (options.textPosition === "center") {
      container.classList.add("b-dics--tp-center");
    } else if (options.textPosition === "bottom") {
      container.classList.add("b-dics--tp-bottom");
    } else if (options.textPosition === "left") {
      container.classList.add("b-dics--tp-left");
    } else if (options.textPosition === "right") {
      container.classList.add("b-dics--tp-right");
    }
  };
  
  
  Dics.prototype._createSlider = function(i, initialMediasContainerWidth) {
    let slider = this._createElement("div", "b-dics__slider");
  
    if (this.options.linesColor) {
      slider.style.color = this.options.linesColor;
    }
  
    slider.style[this.config.positionField] = `${initialMediasContainerWidth * (i + 1)}px`;
  
    this.sliders.push(slider);
  
  
    return slider;
  };
  
  
  /**
   *
   * @param media
   * @param i
   * @param mediaContainer
   * @private
   */
  Dics.prototype._createAltText = function(media, i, mediaContainer) {
    let textContent = media.getAttribute("alt");
    if (textContent) {
      let text = this._createElement("p", "b-dics__text");
  
      if (this.options.arrayBackgroundColorText) {
        text.style.backgroundColor = this.options.arrayBackgroundColorText[i];
      }
      if (this.options.arrayColorText) {
        text.style.color = this.options.arrayColorText[i];
      }
  
      text.appendChild(document.createTextNode(textContent));
  
      mediaContainer.appendChild(text);
    }
  };
  
  
  /**
   *
   * @param media
   * @param mediaContainer
   * @private
   */
  Dics.prototype._rotate = function(media, mediaContainer) {
    media.style.rotate = `-${this.options.rotate}`;
    mediaContainer.style.rotate = this.options.rotate;
  
  };
  
  
  /**
   *
   * @private
   */
  Dics.prototype._removeActiveElements = function() {
    let activeElements = Dics.container.querySelectorAll(".b-dics__slider--active");
  
    for (let activeElement of activeElements) {
      activeElement.classList.remove("b-dics__slider--active");
      utils.removeMultiEvents(Dics.container, ["mousemove", "touchmove"], Dics.prototype._removeActiveElements);
    }
  };
  
  
  /**
   *
   * @param linesOrientation
   * @private
   */
  Dics.prototype._setOrientation = function(linesOrientation) {
    this.config = {};
  
    if (linesOrientation === "vertical") {
      this.config.offsetSizeField = "offsetHeight";
      this.config.offsetPositionField = "offsetTop";
      this.config.sizeField = "height";
      this.config.positionField = "top";
      this.config.clientField = "clientY";
      this.config.pageField = "pageY";
    } else {
      this.config.offsetSizeField = "offsetWidth";
      this.config.offsetPositionField = "offsetLeft";
      this.config.sizeField = "width";
      this.config.positionField = "left";
      this.config.clientField = "clientX";
      this.config.pageField = "pageX";
    }
  
  
  };
  
  
  /**
   *
   * @param event
   * @returns {number}
   * @private
   */
  Dics.prototype._calcPosition = function(event) {
    let containerCoords = this.container.getBoundingClientRect();
    let pixel = !isNaN(event[this.config.clientField]) ? event[this.config.clientField] : event.touches[0][this.config.clientField];
  
    return containerCoords[this.config.positionField] < pixel ? pixel - containerCoords[this.config.positionField] : 0;
  };
  
  
  /**
   *
   * @private
   */
  Dics.prototype._pushSections = function(calcMovePixels, position) {
    // if (this._rePosUnderActualSections(position)) {
    this._setFlex(position, this._isGoingRight);
  
    let section = this.sections[this._activeSlider];
    let postActualSection = this.sections[this._activeSlider + 1];
    let sectionWidth = postActualSection.getBoundingClientRect()[this.config.sizeField] - (calcMovePixels - this.sections[this._activeSlider].getBoundingClientRect()[this.config.sizeField]);
  
  
    section.style.flex = this._isGoingRight === true ? `2 0 ${calcMovePixels}px` : `1 1 ${calcMovePixels}px`;
    postActualSection.style.flex = this._isGoingRight === true ? ` ${sectionWidth}px` : `2 0 ${sectionWidth}px`;
  
    this._setLeftToMedias(this.sections, this.medias);
  
    // }
  };
  
  
  /**
   *
   * @private
   */
  Dics.prototype._setFlex = function(position, isGoingRight) {
    let beforeSumSectionsSize = 0;
  
  
    for (let i = 0; i < this.sections.length; i++) {
      let section = this.sections[i];
      const sectionSize = section.getBoundingClientRect()[this.config.sizeField];
  
      beforeSumSectionsSize += sectionSize;
  
      if ((isGoingRight && position > (beforeSumSectionsSize - sectionSize) && i > this._activeSlider) || (!isGoingRight && position < beforeSumSectionsSize) && i < this._activeSlider) {
        section.style.flex = `1 100 ${sectionSize}px`;
      } else {
        section.style.flex = `0 0 ${sectionSize}px`;
      }
  
    }
  };
  
  
  /**
   *
   * @type {{extend: (function(*=, *, *): *), setMultiEvents: setMultiEvents, removeMultiEvents: removeMultiEvents, getConstructor: (function(*=): string)}}
   */
  let utils = {
  
  
    /**
     * Native extend object
     * @param target
     * @param objects
     * @param options
     * @returns {*}
     */
    extend: function(target, objects, options) {
  
      for (let object in objects) {
        if (objects.hasOwnProperty(object)) {
          recursiveMerge(target, objects[object]);
        }
      }
  
      function recursiveMerge (target, object) {
        for (let property in object) {
          if (object.hasOwnProperty(property)) {
            let current = object[property];
            if (utils.getConstructor(current) === "Object") {
              if (!target[property]) {
                target[property] = {};
              }
              recursiveMerge(target[property], current);
            } else {
              // clearEmpty
              if (options.clearEmpty) {
                if (current == null) {
                  continue;
                }
              }
              target[property] = current;
            }
          }
        }
      }
  
      return target;
    },
  
  
    /**
     * Set Multi addEventListener
     * @param element
     * @param events
     * @param func
     */
    setMultiEvents: function(element, events, func) {
      for (let i = 0; i < events.length; i++) {
        element.addEventListener(events[i], func);
      }
    },
  
  
    /**
     *
     * @param element
     * @param events
     * @param func
     */
    removeMultiEvents: function(element, events, func) {
      for (let i = 0; i < events.length; i++) {
        element.removeEventListener(events[i], func, false);
      }
    },
  
  
    /**
     * Get object constructor
     * @param object
     * @returns {string}
     */
    getConstructor: function(object) {
      return Object.prototype.toString.call(object).slice(8, -1);
    }
  };
  