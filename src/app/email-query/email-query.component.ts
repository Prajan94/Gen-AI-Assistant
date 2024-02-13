import { Component } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-email-query',
  templateUrl: './email-query.component.html',
  styleUrl: './email-query.component.scss'
})
export class EmailQueryComponent {
  response: any;
  aiAuery: any;
  course: any;
  hideBotMessage = false;
  isLoading = false;
  aiMessageArray: any = [];
  humanMessageArray:any = [];
  constructor(
    private commonService : CommonServiceService, private route: Router) { }
    askAi() {
      this.hideBotMessage = true;
      this.isLoading = true;
      this.aiMessageArray.push(this.aiAuery);
      this.commonService.askEmailAI(this.aiAuery).subscribe(item => {
      this.response = item;
      this.humanMessageArray.push(this.response);
      this.isLoading = false;
    }, ((error: any) => {
      this.isLoading = false;
      console.log("Error from flask server --->" + error);
      this.response = "Network Error occured, please try again";
    }))
  }
  gotoHome() {
    this.route.navigateByUrl("home");
  }
}
