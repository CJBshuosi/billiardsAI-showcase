Page({
  data: { available: false, guest: true },
  onShow() { this.setData({ guest: !getApp().globalData.currentUser }) },
})
