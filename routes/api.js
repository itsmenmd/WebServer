var express = require('express');
var router = express.Router();
var mysql = require('mysql');

var con = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "root",
    database: "weather"
})
con.connect(function(err) {
    if (err) throw err;
    console.log("Connected!");
});
/* GET users listing. */
router.get('/', function(req, res, next) {
  res.send('respond with a resource');
});
router.get('/prediction', function(req, res){
    con.query("select * from prediction where id = (select max(id) from prediction)", function(err,result,fields){
        if (err) throw err;
        var a = result[0];
        console.log(a);
        res.json(a);
    });
});
router.get('/condition', function(req, res){
    con.query("select * from weathercondition where id = (select max(id) from weathercondition)", function(err,result,fields){
        if (err) throw err;
        var a = result[0];
        console.log(a);
        res.json(a);
    });
});
router.get('/maxtemp', function(req, res){
    con.query("select * from max_predict where id = (select max(id) from max_predict)", function(err,result,fields){
        if (err) throw err;
        var a = result[0];
        console.log(a);
        res.json(a);
    });
});
router.get('/mintemp', function(req, res){
    con.query("select * from min_predict where id = (select max(id) from min_predict)", function(err,result,fields){
        if (err) throw err;
        var a = result[0];
        console.log(a);
        res.json(a);
    });
});
router.get('/datanow', function(req, res){
    con.query("select * from datanow", function(err,result,fields){
        if (err) throw err;
        var a = result[0];
        console.log(a);
        res.json(a);
    });
});
module.exports = router;
