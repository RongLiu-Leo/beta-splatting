document.addEventListener('DOMContentLoaded', domReady);
    let dics = null;
        function domReady() {
            dics = new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function geometrySceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.geometry')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 2] = 'bicycle';
                        break;
                    case 1:
                        parts[parts.length - 2] = 'treehill';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("geometry-decomposition").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dics.medias = dics._getMedias();
        }

