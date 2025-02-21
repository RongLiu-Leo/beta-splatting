document.addEventListener('DOMContentLoaded', domReady);
    let dicsGeometry = null;
    let dicsLight = null;
        function domReady() {
            dicsGeometry = new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            dicsLight = new Dics({
                container: document.querySelectorAll('.b-dics')[1],
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
            dicsGeometry.medias = dicsGeometry._getMedias();
        }

        function sceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.light')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 3] = 'counter';
                        break;
                    case 1:
                        parts[parts.length - 3] = 'garden';
                        break;
                    case 2:
                        parts[parts.length - 3] = 'kitchen';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("scene").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsLight.medias = dicsLight._getMedias();
        }

        function lightSceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.light')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 1] = 'Diffuse.mp4';
                        break;
                    case 1:
                        parts[parts.length - 1] = 'RGB.mp4';
                        break;
                    case 2:
                        parts[parts.length - 1] = 'Specular.mp4';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("light-decomposition").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsLight.medias = dicsLight._getMedias();
        }

