const SERVER_URL = "http://localhost:5500/api"
const GLOBALS = {
    sim: null,
    gridPoints: 5,
    gridSize: 2.0,
    dP: {px: 2, py: 2, pz: 2, pra: 2, pag: 1, pl: 0, ply: 0, pc: 2, pf: 0, pp: 0},
    loopThickness: 0.01
}

class Sim {
    constructor() {
        this.scene;
        this.camera;
        this.renderer;
        this.controls;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();


        this.coils = [];
        this.selectedCoil;

        this.fieldData;
        this.forceData;
        this.curFrame = 0;
        this.totFrames = 0;
        this.playing = false;
        this.fps = 30;
        this.pl_timeout = null;

        this.vectorF;
        this.coilMeshes = new Map()
        this.forceArr = new Map();
        this.gridHelper;

        this.isDrag = false;
        this.dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
        this.dOffset = new THREE.Vector3();

        this.settings = {
            vectorScale: 1.0,
            showGrid: true,
            showForces: true
        }

        this.init()
        this.setupEventListeners();
        this.animate();
    }

    init() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a);
        
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.01,
            100
        );
        this.camera.position.set(1.5, 1.5, 1.5);
        
        const container = document.getElementById('canvas');
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(GLOBALS.gridSize, GLOBALS.gridSize, GLOBALS.gridSize);
        this.scene.add(directionalLight);
        
        this.gridHelper = new THREE.GridHelper(2 * GLOBALS.gridSize, GLOBALS.gridPoints, 0x444444, 0x222222);
        this.scene.add(this.gridHelper);
        
        const axesHelper = new THREE.AxesHelper(2);
        this.scene.add(axesHelper);


        let temp = localStorage.getItem("sim_id");
        if (temp) {
            fetch(SERVER_URL + `/simulate/${temp}/delete`, { method: 'POST' })
            .then(() => {
                console.log("Deleted old simulation with id:", temp);
                localStorage.removeItem("sim_id");
            })
        }
    }

    setupEventListeners(){
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        })

        // dragging stuff
        this.renderer.domElement.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('mouseup', (e) => this.onMouseUp(e));

        // control stuff
        document.getElementById('add-coil').addEventListener('click', () => this.addCoil());
        document.getElementById('calc-button').addEventListener('click', () => this.calcField());
        document.getElementById('del-button').addEventListener('click', () => this.reset());
        document.getElementById('pprbtn').addEventListener('click', () => this.togglePPR())
        document.getElementById('timeline').addEventListener('input', (e) => this.onTimeLC(e))

        this.setParamListeners()

        document.getElementById('pvss').addEventListener('input', (e) => {
            this.settings.vectorScale = parseFloat(e.target.value);
            document.getElementById('pvs').textContent = e.target.value
            this.updateVectorField()
            this.updateForceArrs()
        })

        document.getElementById('sg').addEventListener('change', (e) => {
            this.settings.showGrid = e.target.checked;
            this.gridHelper.visible = e.target.checked
        })

        document.getElementById('sf').addEventListener('change', (e) => {
            this.settings.showForces = e.target.checked;
            this.forceArr.forEach(arr => arr.visible = e.target.checked)
        })
    }

    setParamListeners() {
        const theps = ['px', 'py', 'pz', 'pra', 'pag', 'pl', 'ply', 'pc', 'pf', 'pp']

        theps.forEach(par => {
            var inp = document.getElementById(`${par}s`)
            var valD = document.getElementById(par)

            inp.addEventListener('input', (e) => {
                var val = parseFloat(e.target.value)
                valD.textContent = val

                if (this.selectedCoil){
                    this.updateCParam()
                }
            })
        })
    }

    updateCParam(){
        if (!this.selectedCoil) return;

        var coil = this.selectedCoil;
        const theps = ['px', 'py', 'pz', 'pra', 'pag', 'pl', 'ply', 'pc', 'pf', 'pp']

        theps.forEach(par => {
            var valD = document.getElementById(par)
            switch(par) {
                case 'px': coil.position[0] = parseFloat(valD.textContent); break;
                case 'py': coil.position[1] = parseFloat(valD.textContent); break;
                case 'pz': coil.position[2] = parseFloat(valD.textContent); break;
                case 'pra': coil.radius = parseFloat(valD.textContent); break;
                case 'pag': coil.angle = parseFloat(valD.textContent); break;
                case 'pl': coil.loops = parseInt(valD.textContent); break;
                case 'ply': coil.layers = parseInt(valD.textContent); break;
                case 'pc': coil.current = parseFloat(valD.textContent); break;
                case 'pf': coil.frequency = parseFloat(valD.textContent); break;
                case 'pp': coil.phase = parseFloat(valD.textContent); break;
            }            
        })
        this.updateCMesh(coil);
    }

    addCoil(){
        var coil = {
            id: this.coils.length > 0 ? Math.max(...this.coils.map(c => c.id)) + 1 : 1,
            position: [0, 0, 0],
            radius: 1.0,
            angle: 0,
            loops: 1,
            layers: 1,
            current: 1.0,
            frequency: 1,
            phase: 0
        }
        this.coils.push(coil);
        this.createCMesh(coil);
        this.updateCList();
        this.updateStatus()
    }

    removeCoil(coil_id){
        this.coils = this.coils.filter(c => c.id !== coil_id)
        this.coils.forEach((c, ind) => c.id = ind + 1)
        var mesh = this.coilMeshes.get(coil_id);
        if (mesh){
            this.scene.remove(mesh);
            mesh.geometry.dispose()
            mesh.material.dispose()
            this.coilMeshes.delete(coil_id)
        }

        const arr = this.forceArr.get(coil_id)
        if (arr){
            this.scene.remove(arr)
            this.forceArr.delete(coil_id)
        }

        if (this.selectedCoil?.id === coil_id) {
            this.selectedCoil = null;
            document.getElementById('coils-params').style.display = 'none'
        }

        this.updateCList();
        this.updateStatus()
        
    }

    selectC(coil){
        this.selectedCoil = coil;

        document.getElementById('coils-params').style.display = 'block'
        document.getElementById('pxs').value = coil.position[0]
        document.getElementById('pys').value = coil.position[1]
        document.getElementById('pzs').value = coil.position[2]
        document.getElementById('pras').value = coil.radius
        document.getElementById('pags').value = coil.angle
        document.getElementById('pls').value = coil.loops
        document.getElementById('plys').value = coil.layers
        document.getElementById('pcs').value = coil.current
        document.getElementById('pfs').value = coil.frequency
        document.getElementById('pps').value = coil.phase

        document.getElementById('px').textContent = coil.position[0].toFixed(GLOBALS.dP.px)
        document.getElementById('py').textContent = coil.position[1].toFixed(GLOBALS.dP.py)
        document.getElementById('pz').textContent = coil.position[2].toFixed(GLOBALS.dP.pz)
        document.getElementById('pra').textContent = coil.radius.toFixed(GLOBALS.dP.pra)
        document.getElementById('pag').textContent = coil.angle.toFixed(GLOBALS.dP.pag)
        document.getElementById('pl').textContent = coil.loops.toFixed(GLOBALS.dP.pl)
        document.getElementById('ply').textContent = coil.layers.toFixed(GLOBALS.dP.ply)
        document.getElementById('pc').textContent = coil.current.toFixed(GLOBALS.dP.pc)
        document.getElementById('pf').textContent = coil.frequency.toFixed(GLOBALS.dP.pf)
        document.getElementById('pp').textContent = coil.phase.toFixed(GLOBALS.dP.pp)

        this.updateCList()

        this.coilMeshes.forEach((mesh,id) => mesh.material.emissive.set((id == coil.id) ? 0x444444:0x000000))
    }

    createCMesh(coil) {
        const geometry = new THREE.TorusGeometry(coil.radius, GLOBALS.loopThickness, 16, 32);
        const material = new THREE.MeshStandardMaterial({
            color: 0x4CAF50,
            metalness: 0.7,
            roughness: 0.3
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(...coil.position);
        mesh.rotation.y = THREE.MathUtils.degToRad(coil.angle);
        mesh.userData.coilId = coil.id;
        
        this.scene.add(mesh);
        this.coilMeshes.set(coil.id, mesh);
    }

    updateCMesh(coil){
        const mesh = this.coilMeshes.get(coil.id)
        if (!mesh) return;

        mesh.position.set(...coil.position);
        mesh.rotation.y = THREE.MathUtils.degToRad(coil.angle);

        let currrad = mesh.geometry.parameters.radius;
        if (Math.abs(currrad - coil.radius) > 0.001){
            mesh.geometry.dispose();
            mesh.geometry = new THREE.TorusGeometry(coil.radius, GLOBALS.loopThickness, 16, 32);
        }
    }

    updateCList() {
        const list = document.getElementById('coil-list');
        list.innerHTML = '';
        
        this.coils.forEach((coil, ind) => {
            const item = document.createElement('div');
            item.className = 'coil-item' + (this.selectedCoil && this.selectedCoil.id === coil.id ? ' selected' : '');
            
            
            item.innerHTML = `
                <div class="coil-header">
                    <span class="coil-name">${ind + 1}</span>
                    <div class="coil-actions">
                        <button class="icon-button" onclick="GLOBALS.sim.selectC(GLOBALS.sim.coils.find(c => c.id === ${coil.id}))">Edit</button>
                        <button class="icon-button" onclick="GLOBALS.sim.removeCoil(${coil.id})">Del</button>
                    </div>
                </div>
                <div style="font-size: 11px; color: #888;">
                    Pos: (${coil.position.map(p => p.toFixed(GLOBALS.dP.px)).join(', ')}) | 
                    R: ${coil.radius.toFixed(GLOBALS.dP.pra)}m | 
                    I: ${coil.current.toFixed(GLOBALS.dP.pc)}A
                </div>
            `;
            
            list.appendChild(item);
        });
    }

    async calcField(){
        if (this.coils.length == 0) {
            alert("Bruh add some coils")
            return;
        }

        document.getElementById('loading').classList.remove('hidden')
        document.getElementById('calc-button').disabled = true;

        try {
            const reqD = {
                coils: this.coils,
                limits: GLOBALS.gridSize,
                points: GLOBALS.gridPoints,
            }

            const res = await fetch(SERVER_URL + '/simulate/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqD)
            })

            if (!res.ok) throw new Error('Calculation failed');

            const data = await res.json()
            var saved = JSON.parse(localStorage.getItem('saved_sims') || '[]');
            saved.push(data.sim_id);
            localStorage.setItem('saved_sims', JSON.stringify(saved));
            this.fieldData = data.B
            this.forceData = data.forces // {coil_id: (frames, 3)})

            this.totFrames = this.fieldData.length
            this.curFrame = 0;
            document.getElementById('totalf').textContent = this.totFrames
            document.getElementById('timeline').max = this.totFrames - 1;

            this.createVectorField();
            this.createForceArrs()

            this.updateFrame(0)
            this.updateStatus()
        } catch (error){
            console.error("Error during field calculation:", error);
            alert("An error occurred during calculation. Please try again.");
        } finally {
            document.getElementById('loading').classList.add('hidden')
            document.getElementById('calc-button').disabled = false;
        }
    }

    createVectorField() {
        if (this.vectorF) {
            this.scene.remove(this.vectorF)
            this.vectorF.geometry.dispose();
            this.vectorF.material.dispose()
        }

        var frame0 = this.fieldData[0]
        var nvects = frame0.length;

        const arrgeo = this.createArrGeometry()

        this.vectorF = new THREE.InstancedMesh(
            arrgeo,
            new THREE.MeshStandardMaterial({
                color: 0xEBCF73,
                metalness: 0.3,
                roughness: 0.5
            }),
            nvects
        )

        this.scene.add(this.vectorF)
    }

    createArrGeometry() {

        const cylHeight = 0.8;
        const coneHeight = 0.2;

        const cylgeo = new THREE.CylinderGeometry(0.02, 0.02, cylHeight, 16);
        const conegeo = new THREE.ConeGeometry(0.05, coneHeight, 16);

        cylgeo.translate(0, cylHeight / 2, 0);
        conegeo.translate(0, cylHeight + coneHeight / 2, 0);

        const geo = new THREE.BufferGeometry();
        const co = conegeo.toNonIndexed()
        const cy = cylgeo.toNonIndexed()

        const poses = new Float32Array([
            ...co.attributes.position.array,
            ...cy.attributes.position.array
        ])
        const normals = new Float32Array([
            ...co.attributes.normal.array,
            ...cy.attributes.normal.array
        ])

        geo.setAttribute('position', new THREE.BufferAttribute(poses, 3))
        geo.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
        return geo;
    }

    createForceArrs() {
        this.forceArr.forEach(arrow => {
            this.scene.remove(arrow);
        });
        this.forceArr.clear();
        
        this.coils.forEach(coil => {
            const arrow = new THREE.ArrowHelper(
                new THREE.Vector3(0, 1, 0),
                new THREE.Vector3(...coil.position),
                0.000001,
                0xff0000,
                0.03,
            );
            arrow.visible = this.settings.showForces;
            this.scene.add(arrow);
            this.forceArr.set(coil.id, arrow);
        });
    }

    updateFrame(frame){
        if (!this.fieldData || !this.forceData) return;

        this.curFrame = frame;
        document.getElementById('currf').textContent = this.curFrame
        document.getElementById('timeline').value = this.curFrame

        this.updateVectorField()
        this.updateForceArrs()
    }

    updateVectorField(){
        if (!this.vectorF || !this.fieldData) return;

        const framD = this.fieldData[this.curFrame]
        var matr = new THREE.Matrix4();
        var quatr = new THREE.Quaternion();
        const upp = new THREE.Vector3(0, 1, 0)

        var visible = 0;

        framD.forEach((p, i) => {
            const [x, y, z, Bx, By, Bz] = p;
            const pos = new THREE.Vector3(x, y, z);
            const dir = new THREE.Vector3(Bx, By, Bz)
            const mg = dir.length() * this.settings.vectorScale;

            if (mg < 1e-20) {
                matr.makeScale(0, 0, 0)
                this.vectorF.setMatrixAt(i, matr)
                return;
            }

            dir.normalize();

            quatr.setFromUnitVectors(upp, dir);
            const sc = Math.min(mg, 0.5);
            matr.compose(pos, quatr, new THREE.Vector3(sc, sc, sc))
            this.vectorF.setMatrixAt(i, matr)

            visible++;
        })

        this.vectorF.instanceMatrix.needsUpdate = true;
        document.getElementById('v-count').textContent = visible
    }

    updateForceArrs(){
        if (!this.forceData) return;

        this.coils.forEach(coil => {
            let arr = this.forceArr.get(coil.id);
            if (!arr) return;

            let forV = this.forceData[String(coil.id)]
            if (!forV || forV[this.curFrame] === undefined) return

            var [Fx, Fy, Fz] = forV[this.curFrame]
            let forc = new THREE.Vector3(Fx, Fy, Fz)
            var mag = forc.length() * this.settings.vectorScale

            if (mag < 1e-20){
                arr.visible = false;
                return
            }

            forc.normalize()
            arr.position.set(...coil.position);
            arr.setDirection(forc)
            arr.setLength(Math.min(mag, 0.5))
            arr.visible = this.settings.showForces;
        })
    }

    togglePPR(){
        this.playing = !this.playing
        let btn = document.getElementById('pprbtn')
        btn.textContent = this.playing ? '\u23F8':'\u23F5'
        btn.classList.toggle('active', this.playing)
        clearTimeout(this.pl_timeout)
        if (this.playing) this.play()
    }

    play(){
        if (!this.playing || !this.fieldData) return
        const fD = 1000 / this.fps

        const stp = () => {
            if (!this.playing) return;

            this.curFrame++; this.curFrame %= this.totFrames;
            this.updateFrame(this.curFrame)

            this.pl_timeout = setTimeout(stp, fD)
        }

        stp()
    }

    onTimeLC(e){
        let f = parseInt(e.target.value)
        this.updateFrame(f);
    }
    
    reset(){
        clearTimeout(this.pl_timeout)
        this.playing = false;
        document.getElementById('pprbtn').textContent = '\u23F5'
        document.getElementById('pprbtn').classList.remove('active')

        this.fieldData = null;
        this.forceData = null;
        this.curFrame = 0;
        this.totFrames = 0;

        if (this.vectorF){
            this.scene.remove(this.vectorF)
            this.vectorF.geometry.dispose()
            this.vectorF.material.dispose()
            this.vectorF = null;
        }

        this.forceArr.forEach(arr => this.scene.remove(arr))
        this.forceArr.clear()

        document.getElementById('currf').textContent = '0'
        document.getElementById('totalf').textContent = '0'
        document.getElementById('timeline').textContent = 0;
        document.getElementById('v-count').textContent = '0'
    }

    onMouseDown(event) {
        if (event.button !== 0) return;
        
        this.updateMousePosition(event);
        
        // Raycast to find coil
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.coilMeshes.values()));
        
        if (intersects.length > 0) {
            const mesh = intersects[0].object;
            const coilId = mesh.userData.coilId;
            const coil = this.coils.find(c => c.id === coilId);
            
            if (coil) {
                this.isDrag = true;
                this.selectC(coil);
                this.controls.enabled = false;
                
                // Calculate drag offset
                const planeIntersect = new THREE.Vector3();
                this.raycaster.ray.intersectPlane(this.dragPlane, planeIntersect);
                this.dOffset.copy(planeIntersect).sub(mesh.position);
            }
        }
    }

    onMouseMove(event){
        this.updateMousePosition(event)

        if (this.isDrag && this.selectedCoil){
            this.raycaster.setFromCamera(this.mouse, this.camera)

            let planeInter = new THREE.Vector3()
            if (this.raycaster.ray.intersectPlane(this.dragPlane, planeInter)){
                let newPos = planeInter.sub(this.dOffset)
                this.selectedCoil.position = [newPos.x, newPos.y, newPos.z]
                this.updateCMesh(this.selectedCoil)

                document.getElementById('px').textContent = newPos.x.toFixed(GLOBALS.dP.px)
                document.getElementById('py').textContent = newPos.y.toFixed(GLOBALS.dP.px)
                document.getElementById('pz').textContent = newPos.z.toFixed(GLOBALS.dP.px)
                document.getElementById('pxs').value = newPos.x.toFixed(GLOBALS.dP.px)
                document.getElementById('pys').value = newPos.y.toFixed(GLOBALS.dP.px)
                document.getElementById('pzs').value = newPos.z.toFixed(GLOBALS.dP.px)
            }
        }
    }

    onMouseUp(e){
        this.isDrag = false;
        this.controls.enabled = true;
    }

    updateMousePosition(e){
        const rect = this.renderer.domElement.getBoundingClientRect()
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    }

    updateStatus(){
        document.getElementById('coil-count').textContent = this.coils.length;
    }

    animate(){
        requestAnimationFrame(() => this.animate())

        this.controls.update()
        this.renderer.render(this.scene, this.camera)
    }
}



document.addEventListener('DOMContentLoaded', () => {
    GLOBALS.sim = new Sim();
    document.getElementById('pxs').min = -GLOBALS.gridSize;
    document.getElementById('pxs').max = GLOBALS.gridSize;
    document.getElementById('pys').min = -GLOBALS.gridSize;
    document.getElementById('pys').max = GLOBALS.gridSize;
    document.getElementById('pzs').min = -GLOBALS.gridSize;
    document.getElementById('pzs').max = GLOBALS.gridSize;
    document.getElementById('pras').min = (GLOBALS.gridSize / GLOBALS.gridPoints);
    document.getElementById('pras').max = GLOBALS.gridSize;
})