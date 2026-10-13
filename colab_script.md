// 1. ล้างตัวจับเวลาเดิมออกก่อน
if (typeof keepAliveTimer !== 'undefined') clearInterval(keepAliveTimer);

function clickColab() {
  const connectButton = document.querySelector('colab-connect-button')?.shadowRoot?.querySelector('#connect');
  
  if (connectButton) {
    connectButton.click();
    console.log(`[Colab Keep-Alive] Clicked connect at: ${new Date().toLocaleTimeString()}`);
  } else {
    window.dispatchEvent(new MouseEvent('mousemove'));
    console.log(`[Colab Keep-Alive] Emulated activity at: ${new Date().toLocaleTimeString()}`);
  }
}

// คลิกทันที 1 ครั้งเพื่อยืนยันว่าทำงานและแสดง Log แรก
clickColab();

// ทำงานซ้ำทุกๆ 300 วินาที (5 นาที)
keepAliveTimer = setInterval(clickColab, 300 * 1000);

console.log("Colab auto-clicker running every 300s (5 mins). To stop: clearInterval(keepAliveTimer)");