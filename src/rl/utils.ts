export function assert(condition, message?: string) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}

export function zeros(n) {
    return new Float64Array(n);
}

// Random numbers utils
var return_v = false;
var v_val = 0.0;
export function gaussRandom() {
    if (return_v) {
        return_v = false;
        return v_val;
    }
    var u = 2 * Math.random() - 1;
    var v = 2 * Math.random() - 1;
    var r = u * u + v * v;
    if (r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2 * Math.log(r) / r);
    v_val = v * c; // cache this
    return_v = true;
    return u * c;
}

export function randf(a: number, b: number) { return Math.random() * (b - a) + a; }
export function randi(a: number, b: number) { return Math.floor(Math.random() * (b - a) + a); }
export function randn(mu: number, std: number) { return mu + gaussRandom() * std; }

export function sig(x: number) {
    // helper function for computing sigmoid
    return 1.0 / (1 + Math.exp(-x));
}


export function maxi(w: Float64Array) {
    // argmax of array w
    var maxv = w[0];
    var maxix = 0;
    for (var i = 1, n = w.length; i < n; i++) {
        var v = w[i];
        if (v > maxv) {
            maxix = i;
            maxv = v;
        }
    }
    return maxix;
}


export function sampleWeighted(p) {
    var r = Math.random();
    var c = 0.0;
    for(var i=0,n=p.length;i<n;i++) {
      c += p[i];
      if(c >= r) { return i; }
    }
    assert(false, 'wtf');
  }