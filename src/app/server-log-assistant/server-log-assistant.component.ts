import { Component } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-server-log-assistant',
  templateUrl: './server-log-assistant.component.html',
  styleUrl: './server-log-assistant.component.scss'
})
export class ServerLogAssistantComponent {
  response: any;
  aiAuery: any;
  course: any;
  isLoading = false;
  hideBotMessage = false;
  errMsg: any;
  aiMessageArray: any = [];
  humanMessageArray:any = [];
  constructor(
    private commonService : CommonServiceService, private route: Router) {
     }
    askAi() {
        this.errMsg = "";
        this.hideBotMessage = true;
        this.isLoading = true;
        this.aiMessageArray.push(this.aiAuery);
        this.commonService.serverAsst(this.aiAuery).subscribe(item => {
        this.response = item;
        this.humanMessageArray.push(item);
        this.isLoading = false;
      }, ((error: any) => {
        this.isLoading = false;
        this.humanMessageArray.push(error.error.text);
        console.log("Error from flask server --->" + error);
        // this.response = "Network Error occured, please try again";
      }))
  }
  gotoHome() {
    this.route.navigateByUrl("home");
  }
}
