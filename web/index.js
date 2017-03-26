document.body.innerHTML =
  "Connecting to server..." 


$(function() {
  var socket = io('http://hackathon:3000');
  socket.on('id', function (event) {
    console.log(event.id);
    document.body.innerHTML = contentUpload();
    uploadListener($('#file'), socket);
  });

});

function uploadListener(input, socket) {
  input.change(function(e) {
    console.log('Uploading file');
    var file = e.target.files[0];
    var stream = ss.createStream();

    // upload a file to the server.
    var socketStream = ss(socket);
    socketStream.emit('file', stream, {size: file.size});
    ss.createBlobReadStream(file).pipe(stream);

    socket.on('file_uploaded', function() {
      document.body.innerHTML = '';
    });

    socket.on('training', function (data) {
      var content = document.body.innerHTML;
      content = content + data.toString();
      document.body.innerHTML = content;
    })
  });
}

function contentUpload() {
  return  '<input id="file" type="file" accept="*">'
}
