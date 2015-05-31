var _ = require('underscore');

module.exports = {
    conv1d: function(input, kernel) {
        return Conv1d(input, kernel);
    },

    conv2d: function(input, kernel) {
        return Conv2D(input, kernel);
    }
};


var Conv2D = function(input, kernel) {
    var hlen = input.length;
    var wlen = input[0].length;
    var result = zerosn(hlen, wlen);
    var wh = kernel.length;
    var ww = kernel[0].length;
    var kernX = Math.floor(wh / 2);
    var kernY = Math.floor(ww / 2);
    var len = input.length + kernel.length - 1;
    for (var k = 0; k < hlen; ++k) {
        for (var m = 0; m < wlen; ++m) {
            var temp = 0;
            for (var i = 0; i < kernel.length; ++i) {
                for (var j = 0; j < kernel[i].length; ++j) {
                    if (k == 2 && m === 0) {}
                    var kernresult = ConvHelp2d(kernel, kernX - j + 1, kernY - i + 1);
                    temp += ConvHelp2d(input, j + k - 1, i - 1 + m) * kernresult;
                }
            }
            result[k][m] = temp;
        }
    }
    return result;
};

var ConvHelp2d = function(data, i, j) {
    if (i < 0 || j < 0)
        return 0;
    if (i === undefined || j === undefined)
        return 0;
    if (i > data.length - 1 || j > data[0].length - 1)
        return 0;
    return data[i][j];
};


//1D convolution
var Conv1d = function(input, kernel) {
    var result = [];
    kernel = _.flatten(kernel);
    var len = input.length + kernel.length - 1;
    for (var i = 0; i < len; ++i) {
        var value = 0;
        for (var j = 0; j <= i; ++j) {
            var first = input[j];
            var res = kernel[i - j];
            if (res === undefined)
                res = 0;
            value += ComputeConv1D(input[j], kernel[i - j]);
        }
        result.push(value);
    }
    return result;
};

//x[i] * h[j-i]
var ComputeConv1D = function(invalue, kervalue) {
    if (invalue === undefined || kervalue === undefined)
        return 0;
    return invalue * kervalue;
};

var zeros = function(size) {
    return Array.apply(null, Array(size)).map(function() {
        return 0;
    });
};
var zerosn = function(size1, size2) {
    var w = zeros(size2);
    return zeros(size1).map(function() {
        return zeros(size2);
    });

};