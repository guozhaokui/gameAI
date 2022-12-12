import { assert, randf, randn, zeros } from "./utils";

export class Mat {
    // 行数
    n = 1;
    // 列数
    d = 1;
    // 权重
    w: Float64Array;
    // d
    dw: Float64Array;

    constructor(n: number, d: number, mu?:number, std?:number) {
        // n is number of rows d is number of columns
        this.n = n;
        this.d = d;
        if(mu&&std){
            Mat.fillRandn(this, mu, std);        
        }else{
            this.w = zeros(n * d);
        }
        this.dw = zeros(n * d);

    }

    get(row: number, col: number) {
        // slow but careful accessor function
        // we want row-major order
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        return this.w[ix];
    }

    set(row: number, col: number, v: number) {
        // slow but careful accessor function
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        this.w[ix] = v;
    }

    setFrom(arr: Float64Array) {
        for (var i = 0, n = arr.length; i < n; i++) {
            this.w[i] = arr[i];
        }
    }

    setColumn(m: Mat, i: number) {
        for (var q = 0, n = m.w.length; q < n; q++) {
            this.w[(this.d * q) + i] = m.w[q];
        }
    }

    toJSON() {
        var json = { n: this.n, d: this.d, w: [] };
        for (let i = 0, num = this.n * this.d; i < num; i++) {
            json.w[i] = this.w[i];
        }
        return json;
    }

    fromJSON(json) {
        this.n = json.n;
        this.d = json.d;
        this.w = zeros(this.n * this.d);
        this.dw = zeros(this.n * this.d);
        for (var i = 0, n = this.n * this.d; i < n; i++) {
            this.w[i] = json.w[i]; // copy over weights
        }
    }

    // fill matrix with random gaussian numbers
    static fillRandn(m:Mat, mu:number, std:number) {
        for (var i = 0, n = m.w.length; i < n; i++) {
            m.w[i] = randn(mu, std);
        }
    }

    static fillRand(m, lo, hi) {
        for (var i = 0, n = m.w.length; i < n; i++) {
            m.w[i] = randf(lo, hi);
        }
    }

    static gradFillConst(m, c) {
        for (var i = 0, n = m.dw.length; i < n; i++) {
            m.dw[i] = c
        }
    }


    static copyMat(b) {
        var a = new Mat(b.n, b.d);
        a.setFrom(b.w);
        return a;
    }

    static copyNet(net) {
        // nets are (k,v) pairs with k = string key, v = Mat()
        var new_net = {};
        for (var p in net) {
            if (net.hasOwnProperty(p)) {
                new_net[p] = Mat.copyMat(net[p]);
            }
        }
        return new_net;
    }

    static updateMat(m, alpha) {
        // updates in place
        for (var i = 0, n = m.n * m.d; i < n; i++) {
            if (m.dw[i] !== 0) {
                m.w[i] += - alpha * m.dw[i];
                m.dw[i] = 0;
            }
        }
    }

    static updateNet(net, alpha) {
        for (var p in net) {
            if (net.hasOwnProperty(p)) {
                Mat.updateMat(net[p], alpha);
            }
        }
    }

    static netToJSON(net) {
        var j = {};
        for (var p in net) {
            if (net.hasOwnProperty(p)) {
                j[p] = net[p].toJSON();
            }
        }
        return j;
    }
    static netFromJSON(j) {
        var net = {};
        for (var p in j) {
            if (j.hasOwnProperty(p)) {
                net[p] = new Mat(1, 1); // not proud of this
                net[p].fromJSON(j[p]);
            }
        }
        return net;
    }
    static netZeroGrads(net) {
        for (var p in net) {
            if (net.hasOwnProperty(p)) {
                var mat = net[p];
                Mat.gradFillConst(mat, 0);
            }
        }
    }

    static netFlattenGrads(net) {
        var n = 0;
        for (var p in net) { if (net.hasOwnProperty(p)) { var mat = net[p]; n += mat.dw.length; } }
        var g = new Mat(n, 1);
        var ix = 0;
        for (var p in net) {
            if (net.hasOwnProperty(p)) {
                var mat = net[p];
                for (var i = 0, m = mat.dw.length; i < m; i++) {
                    g.w[ix] = mat.dw[i];
                    ix++;
                }
            }
        }
        return g;
    }

    // return Mat but filled with random numbers from gaussian
    static RandMat(n:number, d:number, mu, std) {
        var m = new Mat(n, d);
        Mat.fillRandn(m, mu, std);
        //fillRand(m,-std,std); // kind of :P
        return m;
    }

}
