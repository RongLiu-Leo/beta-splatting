document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[2],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function objectSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics.object')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                let parts = image.src.split('/').slice(0, -1);
                switch (idx) {
                    case 0:
                        parts[parts.length-2] = 'bicycle'
                        image.src = parts.join('/')
                        break;
                    case 1:
                        parts[parts.length-2] = 'kitchen'
                        image.src = parts.join('/')
                        break;
                    case 2:
                        parts[parts.length-2] = 'stump'
                        image.src = parts.join('/')
                        break;
                    case 3:
                        parts[parts.length-2] = 'flowers'
                        image.src = parts.join('/')
                        break;
                    case 4:
                        parts[parts.length-2] = 'garden'
                        image.src = parts.join('/')
                        break; 
                    case 5:
                        parts[parts.length-2] = 'truck'
                        image.src = parts.join('/')
                        break; 
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '/gs.png';
                        break;
                    case 1:
                        image.src = image.src + '/ours.png';
                        break;
                    case 2:
                        image.src = image.src + '/sugar.png';
                        break;

                }
            }

            let scene_list = document.getElementById("object-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }
        function meshSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics.mesh')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                let parts = image.src.split('/').slice(0, -1);
                switch (idx) {
                    case 0:
                        parts[parts.length-1] = 'chair'
                        image.src = parts.join('/')
                        break;
                    case 1:
                        parts[parts.length-1] = 'lego'
                        image.src = parts.join('/')
                        break;
                    case 2:
                        parts[parts.length-1] = 'hotdog'
                        image.src = parts.join('/')
                        break;
                    case 3:
                        parts[parts.length-1] = 'dtu24'
                        image.src = parts.join('/')
                        break;
                    case 4:
                        parts[parts.length-1] = 'dtu106'
                        image.src = parts.join('/')
                        break; 
                    case 5:
                        parts[parts.length-1] = 'dtu122'
                        image.src = parts.join('/')
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '/sugar.png';
                        break;
                    case 1:
                        image.src = image.src + '/ours.png';
                        break;
                    case 2:
                        image.src = image.src + '/neus.png';
                        break;

                }
            }

            let scene_list = document.getElementById("mesh-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }
        function rgbNormalEvent(idx) {
            let dics = document.querySelectorAll('.b-dics.object')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                let parts = image.src.split('/');
                switch (idx) {
                    case 0:
                        parts[parts.length-2] = 'normal'
                        image.src = parts.join('/')
                        break;
                    case 1:
                        parts[parts.length-2] = 'rgb'
                        image.src = parts.join('/')
                        break;
                }
            }

            let scene_list = document.getElementById("object-rgb-normal").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function shrunkenFullEvent(idx) {
            let dics = document.querySelectorAll('.b-dics.densify')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                let parts = image.src.split('/');
                switch (idx) {
                    case 0:
                        parts[parts.length-2] = 'shrunken'
                        image.src = parts.join('/')
                        break;
                    case 1:
                        parts[parts.length-2] = 'full'
                        image.src = parts.join('/')
                        break;
                }
            }
            let scene_list = document.getElementById("densify-shrunken-full").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }
